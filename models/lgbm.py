
import re
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from category_encoders import TargetEncoder

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    cross_val_predict
)
from tools.metrics import (
    apply_metrics,
    prep_data_for_metric,
    get_avg_volumes,
)

from tools.postprocessing import postprocess_submission

offset_name = "last_before_3_after_0"


def compute_metrics(preds, lower, upper, y, offset, X, avg_volumes):

    id_cols = ["country", "brand"]

    prepped_X = prep_data_for_metric(X, avg_volumes)

    prepped_X["actuals"] = y
    prepped_X["forecast"] = (preds + 1) * offset
    prepped_X["lower_bound"] = (lower + 1) * offset
    prepped_X["upper_bound"] = (upper + 1) * offset

    return np.mean(abs(prepped_X.groupby(id_cols).apply(apply_metrics)))

def preprocess(X):

    X = X.copy()

    offset = X[offset_name]

    for col in X.columns:
        if re.match(r".*mean|median", col):
            X[col] = (X[col] - offset) / offset

    return X

if __name__ == "__main__":

    full_df = pd.read_csv("data/gx_merged_lags_months.csv")
    volume_features = pd.read_csv("data/volume_features.csv")
    submission_df = pd.read_csv("data/submission_template.csv")
    train_tuples = pd.read_csv("data/train_split.csv")
    valid_tuples = pd.read_csv("data/valid_split.csv")

    full_df = full_df.merge(volume_features, on=["country", "brand"])

    full_df["volume_offset"] = (full_df["volume"] - full_df[offset_name]) / full_df[offset_name]
    full_df = preprocess(full_df)

    test_df = full_df[full_df.test == 1].copy().reset_index(drop=True)

    full_df = full_df[full_df.test == 0]

    train_df = full_df.merge(train_tuples, how="inner").reset_index(drop=True)
    val_df = full_df.merge(valid_tuples, how="inner").reset_index(drop=True)

    # TODO: no need for calculation every time
    avg_volumes = get_avg_volumes()

    to_drop = ["volume", "volume_offset"]
    categorical_cols = ["country", "brand", "therapeutic_area", "presentation", "month_name"]

    # Prep data
    train_x = train_df.drop(columns=to_drop)
    train_y = train_df.volume_offset
    train_offset = train_df[offset_name]

    full_x = full_df.drop(columns=to_drop)
    full_y = full_df.volume_offset
    full_offset = full_df[offset_name]

    val_x = val_df.drop(columns=to_drop)
    val_y = val_df.volume_offset
    val_y_raw = val_df.volume
    val_offset = val_df[offset_name]

    test_x = test_df.drop(columns=to_drop)
    test_offset = test_df[offset_name]

    # Prep pipeline
    te = TargetEncoder(cols=categorical_cols)
    te_residual = TargetEncoder(cols=categorical_cols)
    lgb = LGBMRegressor(
        n_jobs=-1, n_estimators=50, objective="regression_l1"
    )
    lgb_residual = LGBMRegressor(
        n_jobs=-1, n_estimators=50, objective="regression_l1"
    )

    pipe = Pipeline([
        ("te", te),
        ("lgb", lgb)
    ])

    pipe_residual = Pipeline([
        ("te", te_residual),
        ("lgb", lgb_residual)
    ])

    # Fit cv model
    cv_preds = cross_val_predict(pipe, train_x, train_y)
    train_y_residual = np.abs(cv_preds - train_y)

    pipe.fit(train_x, train_y)
    pipe_residual.fit(train_x, train_y_residual)

    preds = pipe.predict(val_x)
    preds_residual = pipe_residual.predict(val_x)

    preds_test = pipe.predict(test_x)
    preds_test_residual = pipe_residual.predict(test_x)

    # bounds = [0, ,0.5, 1, 1.5, 2]
    bounds = [1]

    min_unc = 1e8
    best_upper_bound = 0
    best_lower_bound = 0
    for upper_bound in bounds:
        for lower_bound in list(bounds):

            print(f"Upper bound: {upper_bound}")
            print(f"Lower bound: {lower_bound}")
            metric_pair = compute_metrics(
                    preds=preds,
                    lower=preds - lower_bound * preds_residual,
                    upper=preds + upper_bound * preds_residual,
                    y=val_y_raw,
                    offset=val_offset,
                    X=val_x,
                    avg_volumes=avg_volumes
                )
            print(metric_pair)

            unc_metric = metric_pair.values[1]

            if unc_metric < min_unc:
                min_unc = unc_metric
                best_upper_bound = upper_bound
                best_lower_bound = lower_bound

    print(min_unc)
    print(best_upper_bound)
    print(best_lower_bound)

    # Retrain with full data -> In case of need

    retrain_full_data = False

    if retrain_full_data:

        cv_preds_full = cross_val_predict(pipe, full_x, full_y)
        full_y_residual = np.abs(cv_preds_full - full_y)

        pipe.fit(full_x, full_y)
        pipe_residual.fit(full_x, full_y_residual)

        preds_test = pipe.predict(test_x)
        preds_test_residual = pipe_residual.predict(test_x)

    # submission_df["pred_95_low"] = np.maximum(preds_test - upper_bound * preds_test_residual, 0)
    submission_df["pred_95_low"] = (preds_test - best_lower_bound * preds_test_residual + 1) * test_offset
    submission_df["pred_95_high"] = (preds_test + best_upper_bound * preds_test_residual + 1) * test_offset
    submission_df["prediction"] = (preds_test + 1) * test_offset

    # print(submission_df[submission_df.prediction < 0])
    submission_df = postprocess_submission(submission_df)

    # submission_df.to_csv("submissions/baseline_relative_postprocess.csv", index=False)


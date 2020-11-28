
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sktools.encoders import QuantileEncoder

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    cross_val_predict
)

from models.lgbm import (compute_metrics, preprocess)

from tools.metrics import (
    get_avg_volumes,
)

from tools.postprocessing import postprocess_submission

offset_name = "last_before_3_after_0"


if __name__ == "__main__":

    file_name = "linear_base_08_12_qe"
    save = False
    retrain_full_data = True

    full_df = pd.read_csv("data/gx_merged_lags_months.csv")
    # volume_features = pd.read_csv("data/volume_features.csv")
    submission_df = pd.read_csv("data/submission_template.csv")
    train_tuples = pd.read_csv("data/train_split.csv")
    valid_tuples = pd.read_csv("data/valid_split.csv")

    # full_df = full_df.merge(volume_features, on=["country", "brand"])

    full_df["volume_offset"] = (full_df["volume"] - full_df[offset_name]) / full_df[offset_name]
    full_df = preprocess(full_df)

    test_df = full_df[full_df.test == 1].copy().reset_index(drop=True)

    full_df = full_df[full_df.test == 0]

    train_df = full_df.merge(train_tuples, how="inner").reset_index(drop=True)
    val_df = full_df.merge(valid_tuples, how="inner").reset_index(drop=True)

    # TODO: no need for calculation every time
    avg_volumes = get_avg_volumes()

    to_drop = ["volume", "volume_offset"]
    categorical_cols = [
        "country", "brand", "therapeutic_area", "presentation", "month_name",
        "month_country", "month_presentation", "month_area",
        "month_country_num", "month_presentation_num", "month_area_num",
        "month_month_num"
    ]

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
    te = QuantileEncoder(cols=categorical_cols, quantile=0.5)
    te_residual = TargetEncoder(cols=categorical_cols)

    lgb_residual = LGBMRegressor(
        n_jobs=-1, n_estimators=50, objective="regression_l1"
    )

    pipe_linear = Pipeline([
        ("te", te),
        ("imp", SimpleImputer(strategy="mean")),
        ("sc", StandardScaler()),
        ("lgb", ElasticNet(alpha=.0005))
    ])

    pipe_residual = Pipeline([
        ("te", te_residual),
        ("lgb", lgb_residual)
    ])

    # Fit cv model
    cv_preds = cross_val_predict(pipe_linear, train_x, train_y)
    train_y_residual = np.abs(cv_preds - train_y)

    pipe_linear.fit(train_x, train_y)
    pipe_residual.fit(train_x, train_y_residual)

    train_cols = (train_x.columns)
    coefs = (pipe_linear[-1].coef_)

    coefs_dict = (dict(zip(train_cols, coefs)))

    coefs_dict_clean = {
        key: value for key, value in coefs_dict.items() if value != 0
    }

    coefs_df = (pd.DataFrame
                .from_dict(coefs_dict_clean, orient="index")
                .reset_index()
                .rename(columns={0: "coef", "index": "feature"})
                )

    print(coefs_df)

    preds = pipe_linear.predict(val_x)
    preds_residual = pipe_residual.predict(val_x)

    preds_test = pipe_linear.predict(test_x)
    preds_test_residual = pipe_residual.predict(test_x)

    # bounds = [0, ,0.5, 1, 1.5, 2]
    upper_bounds = [1.2]
    lower_bounds = [0.8]

    min_unc = 1e8
    best_upper_bound = 0
    best_lower_bound = 0
    for upper_bound in upper_bounds:
        for lower_bound in lower_bounds:

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
    print(f"Best lower {best_lower_bound}")
    print(f"Best upper {best_upper_bound}")

    save_val = val_x.copy().loc[:, ["country", "brand", "month_num"]]
    save_val["y"] = val_y_raw
    save_val["lower"] = preds - best_lower_bound * preds_residual
    save_val["upper"] = preds + best_upper_bound * preds_residual
    save_val["preds"] = preds
    save_val["lower_raw"] = (1 + save_val["lower"]) * val_offset
    save_val["upper_raw"] = (1 + save_val["upper"]) * val_offset
    save_val["preds_raw"] = (1 + save_val["preds"]) * val_offset
    if save:
        save_val.to_csv(f"data/blend/val_{file_name}.csv", index=False)

    # Retrain with full data -> In case of need
    if retrain_full_data:

        cv_preds_full = cross_val_predict(pipe_linear, full_x, full_y)
        full_y_residual = np.abs(cv_preds_full - full_y)

        pipe_linear.fit(full_x, full_y)
        pipe_residual.fit(full_x, full_y_residual)

        preds_test = pipe_linear.predict(test_x)
        preds_test_residual = pipe_residual.predict(test_x)

    # submission_df["pred_95_low"] = np.maximum(preds_test - upper_bound * preds_test_residual, 0)
    submission_df["pred_95_low"] = (preds_test - best_lower_bound * preds_test_residual + 1) * test_offset
    submission_df["pred_95_high"] = (preds_test + best_upper_bound * preds_test_residual + 1) * test_offset
    submission_df["prediction"] = (preds_test + 1) * test_offset

    # print(submission_df[submission_df.prediction < 0])
    submission_df = postprocess_submission(submission_df)

    submission_df["pred_95_low"] = np.maximum(submission_df["pred_95_low"], 0)
    submission_df["pred_95_high"] = np.maximum(submission_df["pred_95_high"], 0)
    submission_df["prediction"] = np.maximum(submission_df["prediction"], 0)

    if save:
        submission_df.to_csv(f"submissions/submission_{file_name}.csv", index=False)


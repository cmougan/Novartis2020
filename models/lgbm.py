
import re
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from category_encoders import TargetEncoder, CountEncoder

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
    prepped_X["forecast"] = np.maximum((preds + 1) * offset, 0)
    prepped_X["lower_bound"] = np.maximum((lower + 1) * offset, 0)
    prepped_X["upper_bound"] = np.maximum((upper + 1) * offset, 0)

    return np.mean(abs(prepped_X.groupby(id_cols).apply(apply_metrics)))

def preprocess(X):

    X = X.copy()

    offset = X[offset_name]
    # Channel
    # X["channel_"] = "Mixed"
    # X.loc[X["B"] > 75, "channel_"] = "B"
    # X.loc[X["C"] > 75, "channel_"] = "C"
    # X.loc[X["D"] > 75, "channel_"] = "D"

    # More data for target encoding
    X["month_country"] = X["month_name"] + "_" + X["country"]
    X["month_presentation"] = X["month_name"] + "_" + X["presentation"]
    X["month_area"] = X["month_name"] + "_" + X["therapeutic_area"]

    # Month-num
    X["month_country_num"] = X["month_num"].map(str) + "_" + X["country"]
    X["month_presentation_num"] = X["month_num"].map(str) + "_" + X["presentation"]
    X["month_area_num"] = X["month_num"].map(str) + "_" + X["therapeutic_area"]
    X["month_month_num"] = X["month_num"].map(str) + "_" + X["month_name"]

    # X["presentation_therapeutic"] = X["therapeutic_area"] + "_" + X["presentation"]
    # X["therapeutic_channel"] = X["therapeutic_area"] + "_" + X["channel_"]
    # X["presentation_channel"] = X["presentation"] + "_" + X["channel_"]
    # X["country_channel"] = X["country"] + "_" + X["channel_"]
    # X["brand_channel"] = X["brand"] + "_" + X["channel_"]
    # X["country_presentation"] = X["country"] + "_" + X["presentation"] + "_" + X["month_num"].map(str)

    #
    # categorical_cols_freq = [
    #     "country", "brand", "therapeutic_area", "presentation", "month_name",
    # ]
    #
    # freq_encoder_feats = CountEncoder(cols=categorical_cols_freq).fit_transform(
    #     full_df.loc[:, categorical_cols_freq]
    # )
    #
    # freq_encoder_feats.columns = [
    #     f"{col}_freq" for col in freq_encoder_feats.columns
    # ]
    #
    # X = pd.concat([X, freq_encoder_feats], axis=1)

    for col in X.columns:
        if re.match(r".*mean|median", col):
            X[col] = (X[col] - offset) / offset

        # if re.match(r".*Inf", col):
        #     X.drop(columns=col)

    X["n_channels"] = (X["A"] > 10).astype(int) + \
                      (X["B"] > 10).astype(int) + \
                      (X["C"] > 10).astype(int)
    return X


if __name__ == "__main__":

    file_name = "target_encoders"
    save = False
    retrain_full_data = False

    full_df = pd.read_csv("data/gx_merged_lags_months.csv")
    # volume_features = pd.read_csv("data/volume_features.csv")
    submission_df = pd.read_csv("data/submission_template.csv")
    train_tuples = pd.read_csv("data/train_split.csv")
    valid_tuples = pd.read_csv("data/valid_split.csv")
    #
    # feat_01 = pd.read_csv("data/feat_01.csv")
    #
    # full_df = full_df.merge(
    #     feat_01,
    #     on=["country", "brand", "month_num"],
    #     how="left"
    # )

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
        "month_month_num",
        # "presentation_therapeutic",
        # "therapeutic_channel",
        # "presentation_channel",
        # "country_channel",
        # "brand_channel",
        # "channel_",
        # "country_presentation"
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
    te = TargetEncoder(cols=categorical_cols)
    te_residual = TargetEncoder(cols=categorical_cols)
    lgb = LGBMRegressor(
        n_jobs=-1, n_estimators=50, objective="regression_l1"
    )
    lgb_residual = LGBMRegressor(
        n_jobs=-1, n_estimators=10, objective="regression_l1"
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
        val_x.to_csv(f"data/blend/val_x.csv", index=False)

    # Retrain with full data -> In case of need
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

    submission_df["pred_95_low"] = np.maximum(submission_df["pred_95_low"], 0)
    submission_df["pred_95_high"] = np.maximum(submission_df["pred_95_high"], 0)
    submission_df["prediction"] = np.maximum(submission_df["prediction"], 0)
    if save:
        submission_df.to_csv(f"submissions/submission_{file_name}.csv", index=False)


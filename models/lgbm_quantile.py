
import re
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from category_encoders import TargetEncoder

from sklearn.pipeline import Pipeline
from tools.metrics import (
    get_avg_volumes,
)

from tools.postprocessing import postprocess_submission
from models.lgbm import (compute_metrics, preprocess)

offset_name = "last_before_3_after_1"




if __name__ == "__main__":

    file_name = "quantiles_bgfix"
    save = True
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
    te = TargetEncoder(cols=categorical_cols)
    te_residual = TargetEncoder(cols=categorical_cols)

    lgbms = {}
    pipes = {}
    preds = {}
    preds_test = {}

    for quantile in [0.5, 0.25, 0.75]:

        lgbms[quantile] = LGBMRegressor(
            n_jobs=-1, n_estimators=50, objective="quantile", alpha=quantile,
        )

        pipes[quantile] = Pipeline([
            ("te", te),
            ("lgb", lgbms[quantile])
        ])


        # Fit cv model
        pipes[quantile].fit(train_x, train_y)

        preds[quantile] = pipes[quantile].predict(val_x)
        preds_test[quantile] = pipes[quantile].predict(test_x)

    # bounds = [0, ,0.5, 1, 1.5, 2]
    upper_bounds = [0.8, 1.]
    lower_bounds = [1., 1.2]

    min_unc = 1e8
    best_upper_bound = 0
    best_lower_bound = 0
    for upper_bound in upper_bounds:
        for lower_bound in lower_bounds:

            print(f"Upper bound: {upper_bound}")
            print(f"Lower bound: {lower_bound}")
            metric_pair = compute_metrics(
                    preds=preds[0.5],
                    lower=preds[0.25] * lower_bound,
                    upper=preds[0.75] * upper_bound,
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
    save_val["lower"] = preds[0.25] * best_lower_bound
    save_val["upper"] = preds[0.75] * best_upper_bound
    save_val["preds"] = preds[0.5]
    save_val["lower_raw"] = (1 + save_val["lower"]) * val_offset
    save_val["upper_raw"] = (1 + save_val["upper"]) * val_offset
    save_val["preds_raw"] = (1 + save_val["preds"]) * val_offset
    if save:
        save_val.to_csv(f"data/blend/val_{file_name}.csv", index=False)

    # Retrain with full data -> In case of need
    if retrain_full_data:

        for quantile in [0.5, 0.25, 0.75]:

            print(f"Retraining {quantile}")
            # Fit cv model
            pipes[quantile].fit(full_x, full_y)

            preds_test[quantile] = pipes[quantile].predict(test_x)

    submission_df["pred_95_low"] = (preds_test[0.25] * best_lower_bound + 1) * test_offset
    submission_df["pred_95_high"] = (preds_test[0.75] * best_upper_bound + 1) * test_offset
    submission_df["prediction"] = (preds_test[0.5] + 1) * test_offset

    submission_df = postprocess_submission(submission_df)

    submission_df["pred_95_low"] = np.maximum(submission_df["pred_95_low"], 0)
    submission_df["pred_95_high"] = np.maximum(submission_df["pred_95_high"], 0)
    submission_df["prediction"] = np.maximum(submission_df["prediction"], 0)
    if save:
        submission_df.to_csv(f"submissions/submission_{file_name}.csv", index=False)




import re
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from category_encoders import TargetEncoder

from sklearn.pipeline import Pipeline
from tools.metrics import (
    get_avg_volumes,
)
from tqdm import tqdm

from tools.postprocessing import postprocess_submission
from models.lgbm import (compute_metrics, preprocess)

offset_name = "last_before_3_after_0"


if __name__ == "__main__":

    file_name = "specialist"
    save = False
    retrain_full_data = False

    # THIS IS NOT AS ALWAYS
    full_df_raw = pd.read_csv("data/gx_merged_lags_months.csv")
    submission_df = pd.read_csv("data/submission_template.csv")
    train_tuples = pd.read_csv("data/train_split.csv")
    valid_tuples = pd.read_csv("data/valid_split.csv")
    max_months = pd.read_csv("data/max_months.csv")

    full_df_raw["volume_offset"] = (full_df_raw["volume"] - full_df_raw[offset_name]) / full_df_raw[offset_name]

    full_df_raw = preprocess(full_df_raw)

    for time in (range(1, 24)):

        print("-" * 20)
        print(f"Time {time}")

        specialist = pd.read_csv(f"specialist/vol_{time}.csv")

        full_df = full_df_raw.merge(specialist, on=["country", "brand"])

        test_df = full_df[full_df.test == 1].copy().reset_index(drop=True)

        full_df = full_df[full_df.test == 0]

        test_df = test_df[test_df.month_num >= time].reset_index(drop=True)

        # Only keep elements in full
        test_df = test_df.merge(
            full_df.loc[:, ["country", "brand"]].drop_duplicates(),
            how="inner"
        )

        test_df = test_df.merge(max_months, how='left', on=["country", "brand"])
        test_df = test_df[test_df.max_month == time].reset_index(drop=True)
        test_df = test_df.drop(columns=["max_month", "max_month_1"])

        if test_df.shape[0] == 0:
            continue
        else:
            print("Gonna train")

        submission_df = pd.read_csv("data/submission_template.csv")

        submission_df = submission_df.merge(
            test_df.loc[:, ["country", "brand", "month_num"]].drop_duplicates(),
            how="inner"
        )

        submission_df = submission_df.loc[submission_df.month_num >= time].reset_index(drop=True)

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

        upper_bounds = [1.]
        lower_bounds = [1.]

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

        # Retrain with full data -> In case of need
        if retrain_full_data:

            for quantile in [0.5, 0.25, 0.75]:

                print(f"Retraining {quantile}")
                # Fit cv model
                pipes[quantile].fit(full_x, full_y)

                preds_test[quantile] = pipes[quantile].predict(test_x)

        submission_df["pred_95_low"] = (preds_test[0.25] * 1 + 1) * test_offset
        submission_df["pred_95_high"] = (preds_test[0.75] * 1 + 1) * test_offset
        submission_df["prediction"] = (preds_test[0.5] + 1) * test_offset

        wrong_cond = submission_df.pred_95_low > submission_df.pred_95_high
        aux_low = submission_df.loc[wrong_cond, "pred_95_low"]
        aux_high = submission_df.loc[wrong_cond, "pred_95_high"]

        submission_df.loc[wrong_cond, "pred_95_low"] = aux_high
        submission_df.loc[wrong_cond, "pred_95_high"] = aux_low

        if time == 1:
            full_submission = submission_df
        else:

            print("-" * 25)
            full_submission = full_submission.append(submission_df).reset_index(drop=True)

    full_submission.to_csv(f"specialist/specialist_submission.csv", index=False)


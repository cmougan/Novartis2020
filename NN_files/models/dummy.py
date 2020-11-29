
import re
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from category_encoders import TargetEncoder

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, cross_val_predict
)
from tools.metrics import (
    apply_metrics,
    prep_data_for_metric,
    get_avg_volumes,
    mean_absolute_percentage_error
)

from tools.postprocessing import postprocess_submission


def compute_metrics(preds, lower, upper, y, X, avg_volumes):

    id_cols = ["country", "brand"]

    prepped_X = prep_data_for_metric(X, avg_volumes)

    prepped_X["actuals"] = y
    prepped_X["forecast"] = preds
    prepped_X["lower_bound"] = lower
    prepped_X["upper_bound"] = upper

    return np.mean(abs(prepped_X.groupby(id_cols).apply(apply_metrics)))


if __name__ == "__main__":

    full_df = pd.read_csv("data/gx_merged_lags_months.csv")
    submission_df = pd.read_csv("data/submission_template.csv")
    train_tuples = pd.read_csv("data/train_split.csv")
    valid_tuples = pd.read_csv("data/valid_split.csv")

    test_df = full_df[full_df.test == 1].copy().reset_index(drop=True)

    full_df = full_df[full_df.test == 0]

    train_df = full_df.merge(train_tuples, how="inner").reset_index(drop=True)
    val_df = full_df.merge(valid_tuples, how="inner").reset_index(drop=True)

    # TODO: no need for calculation every time
    avg_volumes = get_avg_volumes()

    median_volumes = (train_df
                      .groupby("month_num")
                      .agg({"volume": "median"})
                      .reset_index()
                      .rename(columns={"volume": "median_volume"})
                      )

    to_drop = ["volume", "month_name"]
    categorical_cols = ["country", "brand", "therapeutic_area", "presentation"]

    train_x = train_df.drop(columns=to_drop)
    train_y = train_df.volume

    val_x = val_df.drop(columns=to_drop)
    val_y = val_df.volume

    test_x = test_df.drop(columns=to_drop)

    train_x = (train_x.merge(median_volumes, how="left", on="month_num"))
    val_x = (val_x.merge(median_volumes, how="left", on="month_num"))
    test_x = (test_x.merge(median_volumes, how="left", on="month_num"))

    # mean_pred = 1e6
    # upper_pred = 2e6
    # lower_pred = 0
    val_x["median_volume"] = 1
    test_x["median_volume"] = 1
    metric_pair = compute_metrics(
            val_x["median_volume"],
            val_x["median_volume"] - 1,
            val_x["median_volume"] + 1,
            val_y,
            val_x,
            avg_volumes
        )
    print(metric_pair)

    submission_df["prediction"] = test_x["median_volume"]
    submission_df["pred_95_low"] = test_x["median_volume"] - 1
    submission_df["pred_95_high"] = test_x["median_volume"] + 1

    submission_df = postprocess_submission(submission_df)


    submission_df.to_csv("submissions/dummy_w_postprocess.csv", index=False)



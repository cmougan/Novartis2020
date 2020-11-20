
import pandas as pd
import numpy as np

from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from tools.simple_metrics import (
    interval_score_loss,
    interval_score_metric,
    interval_score_objective,
    duplicate_df
)



if __name__ == "__main__":

    # Load data and split
    train_raw = pd.read_csv("data/feature_engineered/train_1.csv")
    # test_raw = pd.read_csv("data/feature_engineered/test_1.csv")

    train, val = train_test_split(
        train_raw,
        random_state=42
    )

    # Create duplicated datasets - we'll be able to predict upper and lower bound,
    # one for each of the replications
    train = duplicate_df(train)
    val = duplicate_df(val)

    # Clean data
    to_drop = ['target', 'Cluster', 'brand_group', 'cohort', 'Country']
    train_x = train.drop(columns=to_drop)
    train_y = train.target

    val_x = val.drop(columns=to_drop)
    val_y = val.target

    # Create and fit lgbm - use interval_score_objective
    lgb = LGBMRegressor(objective=interval_score_objective, n_estimators=6000)

    lgb.fit(
        train_x, train_y,
        eval_metric=interval_score_metric,
        eval_set=[(train_x, train_y), (val_x, val_y)],
        verbose=100
    )

    # Predict duplicates
    preds = lgb.predict(val_x)

    # Split lower and upper bounds
    len_real_val = int(len(val_y) / 2)
    lower_bound_preds = preds[len_real_val:]
    upper_bound_preds = preds[:len_real_val]

    # Get real target
    y_real = val_y[len_real_val:]

    # Compute loss
    print(
        interval_score_loss(
            lower_bound_preds,
            upper_bound_preds,
            y_real,
            alpha=0.25
        )
    )

    # 550-ish
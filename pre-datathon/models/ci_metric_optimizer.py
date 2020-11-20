
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

    train_raw = pd.read_csv("data/feature_engineered/train_1.csv")
    # test_raw = pd.read_csv("data/feature_engineered/test_1.csv")


    train, val = train_test_split(
        train_raw,
        random_state=42
    )

    train = duplicate_df(train)
    val = duplicate_df(val)

    to_drop = ['target', 'Cluster', 'brand_group', 'cohort', 'Country']

    train_x = train.drop(columns=to_drop)
    train_y = train.target

    test_x = val.drop(columns=to_drop)
    test_y = val.target

    # objective = partial(ciloss_objective, alpha=0.25, penalization=50)

    alpha = 0.25
    penalization = 100

    lgb = LGBMRegressor(objective=interval_score_objective, n_estimators=5000)

    lgb.fit(
        train_x, train_y,
        eval_metric=interval_score_metric,
        eval_set=[(train_x, train_y), (test_x, test_y)],
        verbose=100
    )

    preds = lgb.predict(test_x)

    len_real_val = int(len(test_y) / 2)

    print(
        interval_score_loss(
            preds[len_real_val:],
            preds[:len_real_val],
            test_y[len_real_val:],
            alpha=0.25
        )
    )

    # 550-ish
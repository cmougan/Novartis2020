

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

from tools.simple_metrics import interval_score_loss


if __name__ == "__main__":

    train = pd.read_csv("data/feature_engineered/train_1.csv")
    # test = pd.read_csv("data/feature_engineered/test_1.csv")

    train, val = train_test_split(
        train,
        random_state=42
    )


    lgb = LGBMRegressor(objective='regression_l1', n_estimators=500)

    to_drop = ['target', 'Cluster', 'brand_group', 'cohort', 'Country']
    train_x = train.drop(columns=to_drop)
    train_y = train.target
    test_x = val.drop(columns=to_drop)
    test_y = val.target

    lgb.fit(train_x, train_y)

    preds = lgb.predict(test_x)

    bounds = [10, 20, 30, 50, 75]

    for bound in bounds:

        print(f"Bound in {bound}")
        print(interval_score_loss(preds - bound, preds + bound, test_y))
        # 670 - ish


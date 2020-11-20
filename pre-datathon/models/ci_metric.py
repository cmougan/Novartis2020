

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split




def interval_score_loss(lower, upper, real, alpha=0.25):
    """
    Taken from: https://stats.stackexchange.com/questions/194660/forecast-accuracy-metric-that-involves-prediction-intervals
    Need to predict lower and upper bounds of interval, use target to assess error.

    :param lower: Lower bound predictions
    :param upper: Upper bound predictions
    :param real: Target
    :param alpha: Alpha in metric in
    :return: Average of interval score loss
    """

    real_lower = 2 * np.abs(real - lower) / alpha
    upper_real = 2 * np.abs(upper - real) / alpha
    upper_lower = np.abs(upper - lower)

    real_lower[real > lower] = 0
    upper_real[real < upper] = 0

    print(f"Lower component {np.sum(real_lower) / len(real):.3f}")
    print(f"Upper component {np.sum(upper_real) / len(real):.3f}")
    print(f"Length component {np.sum(upper_lower) / len(real):.3f}")
    return np.sum(real_lower + upper_real + upper_lower) / len(real)

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


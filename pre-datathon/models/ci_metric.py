

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor




def ci_loss(lower, upper, real, alpha=0.25):

    real_lower = 2 * np.abs(real - lower) / alpha
    upper_real = 2 * np.abs(upper - real) / alpha
    upper_lower = np.abs(upper - lower)

    real_lower[real > lower] = 0
    upper_real[real < upper] = 0

    print(f"Lower component {np.sum(real_lower) / len(real):.3f}")
    print(f"Upper component {np.sum(upper_real) / len(real):.3f}")
    print(f"Length component {np.sum(upper_lower) / len(real):.3f}")
    return np.sum(real_lower + upper_real + upper_lower) / len(real)



train = pd.read_csv("data/feature_engineered/train_1.csv")
test = pd.read_csv("data/feature_engineered/test_1.csv")


lgb = LGBMRegressor(objective='regression_l1')

to_drop = ['target', 'Cluster', 'brand_group', 'cohort', 'Country']
train_x = train.drop(columns=to_drop)
train_y = train.target
test_x = test.drop(columns=to_drop)
test_y = test.target

lgb.fit(train_x, train_y)

preds = lgb.predict(train_x)

bounds = [10, 20, 30, 50, 75]

for bound in bounds:

    print(f"Bound in {bound}")
    print(ci_loss(preds - bound, preds + bound, train_y))


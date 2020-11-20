
import pandas as pd
import numpy as np

from lightgbm import LGBMRegressor
from functools import partial




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


def duplicate_df(df, new_col="upper"):

    df_upper = df.copy()
    df_lower = df.copy()

    df_upper[new_col] = 1
    df_lower[new_col] = 0

    return pd.concat([df_upper, df_lower], axis=0, ignore_index=True)


def ciloss_objective(y, preds, alpha=0.25, penalization=50):

    len_y = int(len(y))
    real_len = int(len_y / 2)

    grad_upper = np.zeros(real_len)
    grad_lower = np.zeros(real_len)
    grad = np.zeros(len_y)
    hess = np.zeros(len_y)

    preds_upper = preds[0:real_len]
    preds_lower = preds[real_len:len_y]
    y_real = y[0:real_len]

    # Upper contribution
    upper_g_lower_cond = preds_upper >= preds_lower
    upper_l_real_cond = preds_upper < y_real
    real_l_lower_cond = y_real < preds_lower
    upper_l_lower_cond = preds_upper < preds_lower

    grad_upper[upper_g_lower_cond] += 1
    grad_upper[upper_l_real_cond] += - 2 / alpha
    grad_upper[upper_l_lower_cond] += - penalization

    grad_lower[upper_g_lower_cond] += -1
    grad_lower[real_l_lower_cond] += 2 / alpha
    grad_lower[upper_l_lower_cond] += penalization

    grad[0:real_len] = grad_upper
    grad[real_len:len_y] = grad_lower

    return grad, hess



train_raw = pd.read_csv("data/feature_engineered/train_1.csv")
test_raw = pd.read_csv("data/feature_engineered/test_1.csv")


train = duplicate_df(train_raw)
test = duplicate_df(test_raw)

to_drop = ['target', 'Cluster', 'brand_group', 'cohort', 'Country']

train_x = train.drop(columns=to_drop)
train_y = train.target

test_x = test.drop(columns=to_drop)
test_y = test.target

objective = partial(ciloss_objective, alpha=0.25, penalization=50)

lgb = LGBMRegressor(objective=ciloss_objective)

lgb.fit(train_x, train_y)

preds = lgb.predict(train_x)

bounds = [10, 20, 30, 50, 75]

for bound in bounds:

    print(f"Bound in {bound}")
    print(ci_loss(preds - bound, preds + bound, train_y))
    print(objective(preds, train_y.values))



import numpy as np
from lightgbm import LGBMRegressor
from sklearn.datasets import load_boston

np.random.seed(42)

def mae_objective(y, preds):

    grad = np.sign(preds - y)
    hess = np.ones(len(y))

    return grad, hess


def mae_metric(y, preds):
    higher_is_better = False
    return 'mae', np.abs(preds - y).mean(), higher_is_better


if __name__ == "__main__":

    X, y = load_boston(return_X_y=True)
    # Using this, it learns
    # cb = LGBMRegressor(objective='regression_l1')
    lgb = LGBMRegressor(objective=mae_objective, n_estimators=100)

    lgb.fit(
        X,
        y,
        eval_metric=mae_metric,
        eval_set=[(X, y),],
        verbose=10
    )

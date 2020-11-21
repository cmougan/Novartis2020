

import numpy as np
from catboost import CatBoostRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error

from tools.catboost_custom import MaeObjective

np.random.seed(42)


if __name__ == "__main__":

    X, y = load_boston(return_X_y=True)
    # Using this, it learns
    cb = CatBoostRegressor(
        loss_function=MaeObjective(),
        # loss_function="MAE",
        eval_metric='MAE'
    )

    cb.fit(
        X,
        y,
    )

    print(mean_absolute_error(cb.predict(X), y))



import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_predict

from tools.simple_metrics import interval_score_loss, OptimizeIntervalScore
from tools.error_metric import error_metric


if __name__ == "__main__":

    train = pd.read_csv("data/feature_engineered/train_1.csv")
    # test = pd.read_csv("data/feature_engineered/test_1.csv")

    train, val = train_test_split(
        train,
        random_state=42
    )

    to_drop = ['target', 'Cluster', 'brand_group', 'cohort', 'Country']
    train_x = train.drop(columns=to_drop)
    train_y = train.target
    test_x = val.drop(columns=to_drop)
    test_y = val.target

    # Create model objects
    # n_estimators = 20
    n_estimators = 500
    # For novartis metric, low n_estimators is better
    # For real metric, high n_estimators is better
    lgb = LGBMRegressor(objective='regression_l1', n_estimators=n_estimators)
    lgb_residual = LGBMRegressor(objective='regression_l1', n_estimators=n_estimators)

    # Obtain predictions of regular cb using cross-validation
    predictions_cv = cross_val_predict(lgb, train_x, train_y)
    # Compute absolute residuals -> | target - prediction |
    y_residual_train = np.abs(predictions_cv - train_y)

    # Fit regular cb to predict target
    lgb.fit(train_x, train_y)
    # Fit residual cb to predict absolute residuals
    lgb_residual.fit(train_x, y_residual_train)

    # Predict everything on test
    preds = lgb.predict(test_x)
    preds_residual = lgb_residual.predict(test_x)

    # Optimize coeficients to minimize loss
    X_ = np.array([preds, preds_residual]).T
    predictions_cv_ul = cross_val_predict(OptimizeIntervalScore(), X_, test_y)

    print(
        interval_score_loss(
            lower=predictions_cv_ul[:, 0],
            upper=predictions_cv_ul[:, 1],
            real=test_y
        )
    )
    # 430 -> optimizer in cross-val


    bounds = [10, 20, 30, 50, 75, 100, 150, 200, 250, 300]

    for bound in bounds:
        print(f"Bound: {bound}")
        print(error_metric(
            test_y,
            preds,
            preds + bound * preds_residual,
            preds - bound * preds_residual)
        )


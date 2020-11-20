

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_predict

from tools.simple_metrics import interval_score_loss


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
    lgb = LGBMRegressor(objective='regression_l1', n_estimators=500)
    lgb_residual = LGBMRegressor(objective='regression_l1', n_estimators=500)

    # Obtain predictions of regular lgb using cross-validation
    predictions_cv = cross_val_predict(lgb, train_x, train_y)
    # Compute absolute residuals -> | target - prediction |
    y_residual_train = np.abs(predictions_cv - train_y)

    # Fit regular lgb to predict target
    lgb.fit(train_x, train_y)
    # Fit residual lgb to predict absolute residuals
    lgb_residual.fit(train_x, y_residual_train)

    # Predict everything on test
    preds = lgb.predict(test_x)
    preds_residual = lgb_residual.predict(test_x)

    betas = [0.1, 0.5, 1, 1.5, 2, 2.5]

    for beta in betas:

        # Compute interval score loss for several elongation of the residuals
        # This is like calibrating the model
        print(f"Beta: {beta}")
        print(interval_score_loss(
            preds - beta * preds_residual,
            preds + beta * preds_residual,
            test_y)
        )
        # 440 - ish


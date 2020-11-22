
####################
#
# This doesn't work. Catboost works with "batches", or on the fly,
# and we need the full vector to compute gradient.
#
#
####################

import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from tools.simple_metrics import (
    interval_score_loss,
    interval_score_metric,
    interval_score_objective,
    duplicate_df
)

from tools.catboost_custom import IntervalScoreObjective, IntervalScoreMetric


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
    print(train.shape)
    train = duplicate_df(train)
    val = duplicate_df(val)

    # Clean data
    to_drop = ['target', 'Cluster', 'brand_group', 'cohort', 'Country']
    train_x = train.drop(columns=to_drop)
    train_y = train.target

    val_x = val.drop(columns=to_drop)
    val_y = val.target

    error = {}
    error_train ={}
    for n_estimators in [10000]:

        print("-" * 20)
        print(n_estimators)

        # Create and fit lgbm - use interval_score_objective
        cb = CatBoostRegressor(loss_function=IntervalScoreObjective(),
                               eval_metric="MAE",
                               n_estimators=n_estimators,
                               verbose=0)


        len_real_train = int(len(train_y) / 2)
        weights = np.ones(len(train_y))
        weights[0:len_real_train] += .0001
        weights[len_real_train:len(train_y)] -= .0001

        cb.fit(
            train_x, train_y, sample_weight=weights
        )

        # Predict duplicates
        preds = cb.predict(val_x)

        # Split lower and upper bounds
        len_real_val = int(len(val_y) / 2)
        lower_bound_preds = preds[len_real_val:]
        upper_bound_preds = preds[:len_real_val]

        # Get real target
        y_real = val_y[len_real_val:]

        error[n_estimators] = interval_score_loss(
                lower_bound_preds,
                upper_bound_preds,
                y_real,
                alpha=0.25
            )
        # Compute loss
        print(
            error[n_estimators]
        )

        # Predict duplicates
        preds_train = cb.predict(train_x)

        # Split lower and upper bounds
        len_real_train = int(len(train_y) / 2)
        lower_bound_preds = preds_train[len_real_train:]
        upper_bound_preds = preds_train[:len_real_train]

        # Get real target
        y_real = train_y[len_real_train:]

        error_train[n_estimators] = interval_score_loss(
            lower_bound_preds,
            upper_bound_preds,
            y_real,
            alpha=0.25
        )

        # Compute loss
        print(
            error_train[n_estimators]
        )

        # 550-ish

    print(error)
    print(error_train)

    # Valid error with n trees
    # {100: 1543.2253280792577, 300: 1412.4471969337526, 500: 1308.6674361817093,
    #  1000: 1227.963743360362, 1500: 1173.3121939442262}

    # Train error with n trees
    # {100: 1383.4605985212715, 300: 1279.224563537674, 500: 1187.3307309950724,
    #  1000: 1117.5252313250912, 1500: 1073.8336250693296}


    # Val - train 10k trees
    # {10000: 903.3864766737887}
    # {10000: 836.0708863713029}


import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from category_encoders import TargetEncoder

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, cross_val_predict
)
from tools.metrics import (
    apply_metrics,
    prep_data_for_metric,
    get_avg_volumes,
    mean_absolute_percentage_error
)


def compute_metrics(preds, lower, upper, y, X, avg_volumes):

    id_cols = ["country", "brand"]

    prepped_X = prep_data_for_metric(X, avg_volumes)

    prepped_X["actuals"] = y
    prepped_X["forecast"] = preds
    prepped_X["lower_bound"] = lower
    prepped_X["upper_bound"] = upper

    return np.mean(abs(prepped_X.groupby(id_cols).apply(apply_metrics)))

if __name__ == "__main__":

    full_df = pd.read_csv("data/gx_merged_lags.csv")
    train_tuples = pd.read_csv("data/train_split.csv")
    valid_tuples = pd.read_csv("data/valid_split.csv")

    full_df = full_df[full_df.test == 0]

    train_df = full_df.merge(train_tuples, how="inner").reset_index(drop=True)
    val_df = full_df.merge(valid_tuples, how="inner").reset_index(drop=True)

    # TODO: no need for calculation every time
    avg_volumes = get_avg_volumes()

    to_drop = ["month_name", "volume"]

    train_x = train_df.drop(columns=to_drop)
    train_y = train_df.volume

    val_x = val_df.drop(columns=to_drop)
    val_y = val_df.volume

    categorical_cols = ["country", "brand", "therapeutic_area", "presentation"]
    te = TargetEncoder(cols=categorical_cols)
    te_residual = TargetEncoder(cols=categorical_cols)
    # imputer = SimpleImputer(strategy="mean")
    lgb = LGBMRegressor(
        n_jobs=-1, n_estimators=100, objective="regression_l1"
    )
    lgb_residual = LGBMRegressor(
        n_jobs=-1, n_estimators=100, objective="regression_l1"
    )

    pipe = Pipeline([
        ("te", te),
        ("lgb", lgb)
    ])

    pipe_residual = Pipeline([
        ("te", te_residual),
        ("lgb", lgb_residual)
    ])

    # TODO: do with group-k-fold
    cv_preds = cross_val_predict(pipe, train_x, train_y)
    train_y_residual = np.abs(cv_preds - train_y)

    pipe.fit(train_x, train_y)
    pipe_residual.fit(train_x, train_y_residual)

    preds = pipe.predict(val_x)
    preds_residual = pipe_residual.predict(val_x)

    train_preds = pipe.predict(train_x)

    # print(np.abs(pipe.predict(train_x))
    # print(f"MAE: {mean_absolute_percentage_error(train_y, pipe.predict(train_x))}")
    # print(f"MAE: {mean_absolute_percentage_error(train_y, train_y.median() + np.zeros(len(train_y)))}")
    # print(f"MAE: {mean_absolute_percentage_error(val_y, np.zeros(len(val_y)))}")
    # print(f"MAE: {mean_absolute_percentage_error(val_y, pipe.predict(val_x))}")
    # print(f"MAE: {mean_absolute_percentage_error(val_y, train_y.median() + np.zeros(len(val_y)))}")
    bounds = [0, 0.01, 0.1, 1, 10, 50, 100,]
    # bounds = [1]

    for bound in bounds:
        print(f"Bound: {bound}")
        print(
            compute_metrics(
                preds,
                preds - bound * preds_residual,
                preds + bound * preds_residual,
                val_y,
                val_x,
                avg_volumes
            )
        )


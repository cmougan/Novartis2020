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

np.random.seed(seed=1234)

def compute_metrics(preds, lower, upper, y, X, avg_volumes):

    id_cols = ["country", "brand"]

    prepped_X = prep_data_for_metric(X, avg_volumes)

    prepped_X["actuals"] = y
    prepped_X["forecast"] = preds
    prepped_X["lower_bound"] = lower
    prepped_X["upper_bound"] = upper

    return np.mean(abs(prepped_X.groupby(id_cols).apply(apply_metrics)))

if __name__ == "__main__":

    full_df = pd.read_csv("data/gx_merged_lags_months.csv")
    submission_df = pd.read_csv("data/submission_template.csv")
    train_tuples = pd.read_csv("data/train_split.csv")
    valid_tuples = pd.read_csv("data/valid_split.csv")

    test_df = full_df[full_df.test == 1].copy().reset_index(drop=True)

    full_df = full_df[full_df.test == 0]

    train_df = full_df.merge(train_tuples, how="inner").reset_index(drop=True)
    val_df = full_df.merge(valid_tuples, how="inner").reset_index(drop=True)
    
    avg_volumes = get_avg_volumes()
    to_drop = ["volume", "month_name"]
    categorical_cols = ["country", "brand", "therapeutic_area", "presentation"]
    
    # Mean and std
    mean_train = np.mean(train_df.volume)
    sd_train = np.std(train_df.volume)
    
    val_x = val_df.drop(columns=to_drop)
    val_y = val_df.volume
    
    val_x['mean_volume_normal'] = np.maximum(np.random.normal(mean_train, sd_train, len(val_x)), 0)
    
    metric_pair = compute_metrics(
            val_x.mean_volume_normal,
            np.maximum(val_x.mean_volume_normal - 0.05*val_x.mean_volume_normal, 0),
            val_x.mean_volume_normal + 0.05*val_x.mean_volume_normal,
            val_y,
            val_x,
            avg_volumes
        )
    
    print(metric_pair)
    
    total_obs_test = len(submission_df)
    prediction = np.maximum(np.random.normal(mean_train, sd_train, total_obs_test), 0)
    pred_95_low = np.maximum(prediction - 0.05*prediction, 0)
    pred_95_high = prediction + 0.05*prediction
    
    submission_df.pred_95_low = pred_95_low
    submission_df.prediction = prediction
    submission_df.pred_95_high = pred_95_high
    
    
    submission_df.to_csv("submissions/baseline_normal.csv", index=False)

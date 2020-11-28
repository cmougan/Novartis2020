
import numpy as np
import pandas as pd

from tools.metrics import (
    prep_data_for_metric,
    apply_metrics,
    get_avg_volumes
)

from tools.postprocessing import postprocess_submission

def compute_metrics_raw(preds, lower, upper, y, X, avg_volumes):

    id_cols = ["country", "brand"]

    prepped_X = prep_data_for_metric(X, avg_volumes)

    prepped_X["actuals"] = y
    prepped_X["forecast"] = preds
    prepped_X["lower_bound"] = lower
    prepped_X["upper_bound"] = upper

    return np.mean(abs(prepped_X.groupby(id_cols).apply(apply_metrics)))

offset_name = "last_before_3_after_0"


if __name__ == "__main__":

    file_name = "blend_simple"
    save = False

    sub_te = pd.read_csv("submissions/submission_target_encoders.csv")
    sub_linear = pd.read_csv("submissions/submission_linear_base.csv")
    sub_quantile = pd.read_csv("submissions/submission_quantiles.csv")

    val_te = pd.read_csv("data/blend/val_target_encoders.csv")
    val_linear = pd.read_csv("data/blend/val_linear_base.csv")
    val_quantile = pd.read_csv("data/blend/val_quantiles.csv")
    val_x = pd.read_csv("data/blend/val_x.csv")

    val_blend = val_te.copy()
    sub_blend = sub_te.copy()

    alpha = 0.5

    for col in ["preds_raw", "lower_raw", "upper_raw"]:
        # (alpha * val_te[col] / 2) + \
        val_blend[col] = (alpha * val_quantile[col]) + \
                         (1 - alpha) * val_linear[col]

    for col in ["pred_95_low", "prediction", "pred_95_high"]:
        # (alpha * sub_te[col] / 2) + \
        sub_blend[col] = (alpha * sub_quantile[col] ) + \
                         (1 - alpha) * sub_linear[col]




    metrics = compute_metrics_raw(
        preds=val_blend["preds_raw"],
        lower=val_blend["lower_raw"],
        upper=val_blend["upper_raw"],
        y=val_blend["y"],
        X=val_x,
        avg_volumes=get_avg_volumes()
    )
    print(metrics)

    submission_df = postprocess_submission(sub_blend)

    submission_df["pred_95_low"] = np.maximum(submission_df["pred_95_low"], 0)
    submission_df["pred_95_high"] = np.maximum(submission_df["pred_95_high"], 0)
    submission_df["prediction"] = np.maximum(submission_df["prediction"], 0)
    if save:
        submission_df.to_csv(f"submissions/submission_{file_name}.csv", index=False)



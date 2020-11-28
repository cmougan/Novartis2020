
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

    file_name_sv = "blend_simple"
    save = False

    submissions = {}
    vals = {}
    alphas = {}

    file_names = ["linear_base", "linear_base_08_12_qe", "quantiles"]

    for file_name in file_names:

        submissions[file_name] = pd.read_csv(f"submissions/submission_{file_name}.csv")
        vals[file_name] = pd.read_csv(f"data/blend/val_{file_name}.csv")

    val_blend = vals[file_names[0]].copy()
    sub_blend = submissions[file_names[0]].copy()
    val_x = pd.read_csv("data/blend/val_x.csv")

    alphas["linear_base"] = 0.4
    alphas["linear_base_08_12_qe"] = 0.4
    alphas["quantiles"] = 0.2

    for col in ["preds_raw", "lower_raw", "upper_raw"]:
        acum = vals[file_names[0]][col] * 0
        for file_name in file_names:
            vals[file_name][col] = np.maximum(vals[file_name][col], 0)
            acum += (alphas[file_name] * vals[file_name][col])
        val_blend[col] = acum

    for col in ["pred_95_low", "prediction", "pred_95_high"]:
        acum = submissions[file_names[0]][col] * 0
        for file_name in file_names:
            submissions[file_name][col] = np.maximum(submissions[file_name][col], 0)
            acum += (alphas[file_name] * submissions[file_name][col])
        sub_blend[col] = acum

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
        submission_df.to_csv(
            f"submissions/submission_{file_name_sv}.csv",
            index=False
        )



import pandas as pd
import numpy as np


def postprocess_submission(submission_df, solve_submission_issues=True):

    join_on = ["country", "brand", "month_num"]
    keep = join_on + ["volume"]

    df_vol = pd.read_csv("../data/gx_volume.csv").loc[:, keep]

    both_ds = submission_df.merge(
        df_vol,
        on=join_on,
        how="left",
    )

    both_ds.loc[both_ds["volume"].notnull(), "prediction"] = both_ds[both_ds["volume"].notnull()]["volume"].values
    both_ds.loc[both_ds["volume"].notnull(), "pred_95_high"] = both_ds[both_ds["volume"].notnull()]["volume"].values + 0.01
    both_ds.loc[both_ds["volume"].notnull(), "pred_95_low"] = both_ds[both_ds["volume"].notnull()]["volume"].values - 0.01

    final_cols = join_on + ["pred_95_low", "prediction", "pred_95_high"]

    final_df =  both_ds.loc[:, final_cols]

    if solve_submission_issues:

        if (final_df.pred_95_low > final_df.pred_95_high).any():
            raise("Stop please, upper < lower")

        cond_lower_mean = final_df.pred_95_low > final_df.prediction
        if cond_lower_mean.any():
            print("Solving lower > mean")
            final_df.loc[cond_lower_mean, "prediction"] = \
                final_df.loc[cond_lower_mean, "pred_95_low"] + 0.01

        cond_upper_mean = final_df.prediction > final_df.pred_95_high
        if cond_upper_mean.any():
            print("Solving upper < mean")
            final_df.loc[cond_upper_mean, "prediction"] = \
                final_df.loc[cond_upper_mean, "pred_95_high"] - 0.01


    return final_df
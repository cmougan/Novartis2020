import pandas as pd
import numpy as np


def postprocess_submission(submission_df):

    join_on = ["country", "brand", "month_num"]
    keep = join_on + ["volume"]

    df_vol = pd.read_csv("data/gx_volume.csv").loc[:, keep]

    both_ds = submission_df.merge(
        df_vol,
        on=join_on,
        how="left",
    )

    both_ds.loc[both_ds["volume"].notnull(), "prediction"] = both_ds[both_ds["volume"].notnull()]["volume"].values
    both_ds.loc[both_ds["volume"].notnull(), "pred_95_high"] = both_ds[both_ds["volume"].notnull()]["volume"].values + 0.01
    both_ds.loc[both_ds["volume"].notnull(), "pred_95_low"] = both_ds[both_ds["volume"].notnull()]["volume"].values - 0.01

    final_cols = join_on + ["pred_95_low", "prediction", "pred_95_high"]

    return both_ds.loc[:, final_cols]

import pandas as pd
from lightgbm import LGBMRegressor
from category_encoders import TargetEncoder

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
from tools.metrics import apply_metrics, custom_metric, uncertainty_metric



def compute_metric(X, y, model):

    X = X.copy()
    preds = model.predict(X)

    id_cols = ["country", "brand"]

    X = X.rename(columns={})
    X[""]

    X.groupby(id_cols).apply(apply_metrics)


if __name__ == "__main__":

    full_df = pd.read_csv("data/gx_merged.csv")

    train_df, val_df = train_test_split(full_df)

    to_drop = ["month_name", "volume"]

    train_x = train_df.drop(columns=to_drop)
    train_y = train_df.volume

    val_x = val_df.drop(columns=to_drop)
    val_y = val_df.volume

    categorical_cols = ["country", "brand", "therapeutic_area", "presentation"]
    te = TargetEncoder(cols=categorical_cols)
    # imputer = SimpleImputer(strategy="mean")
    lgb = LGBMRegressor(n_jobs=-1, n_estimators=100, objective="regression_l1")

    pipe = Pipeline([
        ("te", te),
        ("lgb", lgb)
    ])

    pipe.fit(train_x, train_y)

    print(pipe.predict(val_x))

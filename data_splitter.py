import pandas as pd
import numpy as np

np.random.seed(0)

df = pd.read_csv("data/gx_merged.csv")

df = df[df.test == 0]

df["count_brand"] = df["country"].astype(str) + "-" + df["brand"]


lista = df["count_brand"].unique()
df["count_brand"].nunique()

split = int(1078 * 0.75)

len(lista[:split])


train_split = df[df["count_brand"].isin(lista[:split])]
valid_split = df[~df["count_brand"].isin(lista[:split])]

train_split = train_split[["country", "brand"]]
valid_split = valid_split[["country", "brand"]]


train_split.drop_duplicates()

train_split.drop_duplicates().to_csv("data/train_split.csv", index=False)
valid_split.drop_duplicates().to_csv("data/valid_split.csv", index=False)

import pandas as pd
import numpy as np

np.random.seed(0)

df = pd.read_csv("data/gx_merged.csv")

# Take out test
df = df[df.test == 0]

# Create our unique index variable
df["count_brand"] = df["country"].astype(str) + "-" + df["brand"]

# Unique index
lista = df["count_brand"].unique()
df["count_brand"].nunique()

# Get the ones that have not 24months
a = pd.DataFrame(df.groupby(["country", "brand"]).month_num.max()).reset_index()
a = a[a.month_num < 23]
a["count_brand"] = a["country"].astype(str) + "-" + a["brand"]
deformed = a.count_brand.unique()

buenos = list(set(lista) - set(list(deformed)))

split = int(len(buenos) * 0.75)
split_train_list = buenos[:split]
split_valid_list = buenos[split:]


len(split_train_list)

len(split_valid_list)

train_split = df[df["count_brand"].isin(split_train_list)]
valid_split = df[df["count_brand"].isin(split_valid_list)]

train_split = train_split[["country", "brand"]]
valid_split = valid_split[["country", "brand"]]


train_split.drop_duplicates().to_csv("data/train_split_noerror.csv", index=False)
valid_split.drop_duplicates().to_csv("data/valid_split.csv", index=False)

split_train_split_deformed = list(set((split_train_list + list(deformed))))

train_split = df[df["count_brand"].isin(split_train_split_deformed)]


train_split = train_split[["country", "brand"]]


train_split.drop_duplicates().to_csv("data/train_split.csv", index=False)

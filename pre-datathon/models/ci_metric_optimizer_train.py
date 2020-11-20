
import pandas as pd
import numpy as np

from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import lightgbm




def ci_loss(lower, upper, real, alpha=0.25):

    real_lower = 2 * np.abs(real - lower) / alpha
    upper_real = 2 * np.abs(upper - real) / alpha
    upper_lower = np.abs(upper - lower)

    real_lower[real > lower] = 0
    upper_real[real < upper] = 0

    print(f"Lower component {np.sum(real_lower) / len(real):.3f}")
    print(f"Upper component {np.sum(upper_real) / len(real):.3f}")
    print(f"Length component {np.sum(upper_lower) / len(real):.3f}")
    return np.sum(real_lower + upper_real + upper_lower) / len(real)


def duplicate_df(df, new_col="upper"):

    df_upper = df.copy()
    df_lower = df.copy()

    df_upper[new_col] = 1
    df_lower[new_col] = 0

    return pd.concat([df_upper, df_lower], axis=0, ignore_index=True)


def ciloss_metric(preds, data):
    y = data.get_label()
    is_higher_better = False
    return 'ciloss', ciloss_metric_(y, preds), is_higher_better


def ciloss_metric_(y, preds):

    len_y = round(len(y))
    real_len_ = round(len_y / 2)

    preds_upper_ = preds[:real_len_]
    preds_lower_ = preds[real_len_:]
    y_real = y[0:real_len_]
    acum = np.zeros(real_len_)

    upper_g_lower_cond = preds_upper_ >= preds_lower_
    upper_l_real_cond = preds_upper_ < y_real
    real_l_lower_cond = y_real < preds_lower_
    upper_l_lower_cond = preds_upper_ < preds_lower_

    acum[upper_g_lower_cond] += abs(preds_upper_ - preds_lower_)[upper_g_lower_cond]
    acum[upper_l_real_cond] += (2 / alpha) * abs(preds_upper_ - y_real)[upper_l_real_cond]
    acum[real_l_lower_cond] += (2 / alpha) * abs(preds_lower_ - y_real)[real_l_lower_cond]
    acum[upper_l_lower_cond] += penalization * abs(preds_upper_ - preds_lower_)[upper_l_lower_cond]

    return np.sum(acum) / real_len_


def ciloss_init_score(y):
    p = y.mean()
    return p

def ciloss_objective(preds, train_data):
    y = train_data.get_label()
    return ciloss_objective_(y, preds)

# def ciloss_objective(preds, train_data):
#     y = train_data.get_label()
#     grad = np.sign(y - preds)
#     hess = grad * 0
#     return grad, hess



def ciloss_objective_(y, preds):

    len_y = round(len(y))
    real_len_ = round(len_y / 2)

    grad_upper = np.zeros(real_len_)
    grad_lower = np.zeros(real_len_)
    grad = np.zeros(len_y)
    hess = np.zeros(len_y)

    preds_upper = preds[0:real_len_]
    preds_lower = preds[real_len_:len_y]
    y_real = y[0:real_len_]

    # Upper contribution
    upper_g_lower_cond = preds_upper >= preds_lower
    upper_l_real_cond = preds_upper <= y_real
    real_l_lower_cond = y_real <= preds_lower
    upper_l_lower_cond = preds_upper < preds_lower

    grad_upper[upper_g_lower_cond] += 1
    grad_upper[upper_l_real_cond] += - 2 / alpha
    grad_upper[upper_l_lower_cond] += - penalization

    grad_lower[upper_g_lower_cond] += -1
    grad_lower[real_l_lower_cond] += 2 / alpha
    grad_lower[upper_l_lower_cond] += penalization

    grad[0:real_len_] = grad_upper
    grad[real_len_:len_y] = grad_lower

    # grad = grad / real_len_
    grad = grad * 10

    return grad, hess + 1



train_raw = pd.read_csv("data/feature_engineered/train_1.csv")
# test_raw = pd.read_csv("data/feature_engineered/test_1.csv")


train, val = train_test_split(
    train_raw,
    random_state=42
)
print(train.shape)
print(val.shape)

train = duplicate_df(train)
val = duplicate_df(val)
print(train.shape[0])
print(val.shape[0])
print(train.shape[0] / 2)
print(val.shape[0] / 2)


to_drop = ['target', 'Cluster', 'brand_group', 'cohort', 'Country']

train_x = train.drop(columns=to_drop)
train_y = train.target

test_x = val.drop(columns=to_drop)
test_y = val.target

len_y = len(train_y)
len_real = int(len(train_y) / 2)
init_train = np.zeros(len_y) + train_y.median()
init_train[0:len_real] += 100
init_train[len_real:len_y] += -100


len_y_val = len(test_y)
len_real_val = int(len(test_y) / 2)
init_val = np.zeros(len_y_val) + train_y.median()
init_val[0:len_real_val] += 100
init_val[len_real_val:len_y_val] += -100


fit = lightgbm.Dataset(
    train_x, train_y,
    init_score=init_train
)
val = lightgbm.Dataset(
    test_x, test_y,
    reference=fit,
    init_score=init_val
)

alpha = 0.25
penalization = 1000

model = lightgbm.train(
    params={
        # 'learning_rate': 0.1,
        # 'metric': 'l1',
    },
    train_set=fit,
    num_boost_round=1000,
    valid_sets=(fit, val),
    valid_names=('fit', 'val'),
    verbose_eval=100,
    feval=ciloss_metric,
    fobj=ciloss_objective,
)


preds = (model.predict(test_x))

print(
    ci_loss(
        preds[len_real_val:] + init_val[len_real_val:],
        preds[:len_real_val] + init_val[:len_real_val],
        test_y[len_real_val:],
        alpha=0.25
    )
)
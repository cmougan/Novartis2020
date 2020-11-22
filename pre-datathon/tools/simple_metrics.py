
import pandas as pd
import numpy as np

from tools.constants import ALPHA, PENALIZATION



def interval_score_loss(lower, upper, real, alpha=0.25):
    """
    Taken from: https://stats.stackexchange.com/questions/194660/forecast-accuracy-metric-that-involves-prediction-intervals
    Need to predict lower and upper bounds of interval, use target to assess error.

    :param lower: Lower bound predictions
    :param upper: Upper bound predictions
    :param real: Target
    :param alpha: Alpha in metric in
    :return: Average of interval score loss
    """

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
    """
    Duplicates a df to be able to create lower-upper structure
    :param df: Dataframe to duplicate
    :param new_col: Name of the new column that helps distinguishing
    :return: Duplicated dataframe
    """

    df_upper = df.copy()
    df_lower = df.copy()

    df_upper[new_col] = 1
    df_lower[new_col] = 0

    return pd.concat([df_upper, df_lower], axis=0, ignore_index=True)


def interval_score_metric(y, preds):
    """
    Metric used in lightgbm
    :param y: Duplicated target
    :param preds: Predictions of upper and lower bounds:
     First half of the predictions are for upper bounds.
     Second half of the predictions are for lower bounds.
    :return: Tuple to be consumed by lgbm
    """

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
    acum[upper_l_real_cond] += (2 / ALPHA) * abs(preds_upper_ - y_real)[upper_l_real_cond]
    acum[real_l_lower_cond] += (2 / ALPHA) * abs(preds_lower_ - y_real)[real_l_lower_cond]
    acum[upper_l_lower_cond] += PENALIZATION * abs(preds_upper_ - preds_lower_)[upper_l_lower_cond]

    metric = np.sum(acum) / real_len_
    return 'ciloss', metric, False


def interval_score_objective(y, preds):
    """
    Computes gradient and hessian for lgbm to consume this as an objective.
    :param y:
    :param preds:
    :return: gradient, hessian tuple:
        gradient is calculated properly
        we sum 1 to hessian to have more stable optimization
    """

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

    grad_upper += 1
    grad_upper[upper_l_real_cond] += - 2 / ALPHA
    grad_upper[upper_l_lower_cond] += - PENALIZATION

    grad_lower += -1
    grad_lower[real_l_lower_cond] += 2 / ALPHA
    grad_lower[upper_l_lower_cond] += PENALIZATION

    grad[0:real_len_] = grad_upper
    grad[real_len_:len_y] = grad_lower

    # grad = grad / real_len_
    grad = grad * 10

    return grad, hess + 1
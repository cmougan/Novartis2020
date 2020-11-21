
import numpy as np


def error_metric(real, prediction, upper_bound, lower_bound):

  real_tilde = np.maximum(real, 1)

  err_pred = np.abs(real - prediction)
  err_upper = np.abs(real - upper_bound)
  err_lower = np.abs(real - lower_bound)
  general_term =  (err_pred + err_lower + err_upper) / real_tilde

  relative_upper = (1.5 + (2 * err_upper / real_tilde))
  relative_lower = (1.5 + (2 * err_lower / real_tilde))

  relative_error = np.ones(len(real))

  relative_error[real > upper_bound] = relative_upper[real > upper_bound]
  relative_error[real < lower_bound] = relative_lower[real < lower_bound]

  print(f"Relative term {np.mean(relative_error)}")
  print(f"General term {np.mean(general_term)}")

  return np.mean(relative_error * general_term)


# real = np.zeros(100)
# prediction = real + 1
# upper_bound = prediction + np.arange(100)
# lower_bound = prediction - np.arange(100)

# print(error_metric(real, prediction, upper_bound, lower_bound))
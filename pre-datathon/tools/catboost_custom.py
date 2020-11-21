
import numpy as np

from tools.simple_metrics import (
    interval_score_objective,
    interval_score_metric
)


class IntervalScoreMetric(object):

    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        # approxes is list of indexed containers (containers with only __len__ and __getitem__ defined), one container
        # per approx dimension. Each container contains floats.
        # weight is one dimensional indexed container.
        # target is float.

        # weight parameter can be None.
        # Returns pair (error, weights sum)

        weight_sum = 1.0

        target_np = np.array([target_ for target_ in target])
        approx_np = np.array([approx for approx in approxes])

        error_sum = interval_score_metric(target_np, approx_np)[1]

        return error_sum, weight_sum


class IntervalScoreObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        # approxes, targets, weights are indexed containers of floats
        # (containers which have only __len__ and __getitem__ defined).
        # weights parameter can be None.
        #
        # To understand what these parameters mean, assume that there is
        # a subset of your dataset that is currently being processed.
        # approxes contains current predictions for this subset,
        # targets contains target values you provided with the dataset.
        #
        # This function should return a list of pairs (der1, der2), where
        # der1 is the first derivative of the loss function with respect
        # to the predicted value, and der2 is the second derivative.

        assert len(targets) == len(approxes)

        target_np = np.array([target for target in targets])
        approx_np = np.array([approx for approx in approxes])

        grad, hess = interval_score_objective(target_np, approx_np)
        hess = np.zeros(len(target_np))

        result = []
        for index in range(len(targets)):
            der1 = -grad[index]
            der2 = hess[index]
            result.append((der1, der2))

        return result


class MaeObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        # approxes, targets, weights are indexed containers of floats
        # (containers which have only __len__ and __getitem__ defined).
        # weights parameter can be None.
        #
        # To understand what these parameters mean, assume that there is
        # a subset of your dataset that is currently being processed.
        # approxes contains current predictions for this subset,
        # targets contains target values you provided with the dataset.
        #
        # This function should return a list of pairs (der1, der2), where
        # der1 is the first derivative of the loss function with respect
        # to the predicted value, and der2 is the second derivative.

        assert len(targets) == len(approxes)

        result = []
        for index in range(len(targets)):
            der1 = 1 if approxes[index] < targets[index] else -1
            der2 = 0

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))

        return result


# This is actually slower!!!

class MaeVectorObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        # approxes, targets, weights are indexed containers of floats
        # (containers which have only __len__ and __getitem__ defined).
        # weights parameter can be None.
        #
        # To understand what these parameters mean, assume that there is
        # a subset of your dataset that is currently being processed.
        # approxes contains current predictions for this subset,
        # targets contains target values you provided with the dataset.
        #
        # This function should return a list of pairs (der1, der2), where
        # der1 is the first derivative of the loss function with respect
        # to the predicted value, and der2 is the second derivative.

        assert len(targets) == len(approxes)

        target_np = np.array([target for target in targets])
        approx_np = np.array([approx for approx in approxes])

        grad = np.sign(approx_np - target_np)
        hess = np.zeros(len(target_np))

        result = []
        for index in range(len(targets)):
            der1 = -grad[index]
            der2 = hess[index]
            result.append((der1, der2))

        return result
import re
# import numpy as np
import torch

NaturalSortRegex = re.compile('([0-9]+)')


# for sorting with natural ordering
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(NaturalSortRegex, s)]


def rmse(predictions, targets):
    """
    RMSE loss for numpy
    :param predictions: Numpy array of predictions
    :param targets: Numpy array of targets
    :return: RMSE loss
    """
    return torch.sqrt(((predictions - targets) ** 2).mean())

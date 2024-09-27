import numpy as np
from functools import cache


def anomaly_iqr(dataset):
    """
    **This Uses Interquatile range to identify the outliers.**\n
    lower_bound = q1 - 1.5IQR \n
    higher_bound = q3 + 1.5IQR \n
    """
    q1, q3 = np.percentile(dataset, [25, 75])
    interquatile_range = q3 - q1
    lower_bound = q1 - 1.5 * interquatile_range
    higher_bound = q3 + 1.5 * interquatile_range
    return (lower_bound, higher_bound)


# +++++++++++++++++++++ #
# Module for evaluation #
# +++++++++++++++++++++ #

import numpy as np

def rmspe(y_true, y_pred):
    """
    """
    return np.sqrt(np.mean(((y_true-y_pred)/y_true) ** 2))

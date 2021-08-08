
# Module for pre-processing functions

import numpy as np

def compute_WAP(book_df):
    """
    """
    bidP1, askP1, bidQ1, askQ1 = [book_df[c] for c in ["bid_price1", "ask_price1", "bid_size1", "ask_size1"]]
    book_df.loc[:, "WAP"] = (bidP1*askQ1 + askP1*bidQ1)/(bidQ1 + askQ1)
    return

def compute_lnret(data, varname):
    """
    """
    data.loc[: , varname+"_lnret"] = np.log((data[varname]/data[varname].shift(1)))
    return data

def realized_vol(ln_ret_series):
    """
    """
    return np.sqrt(np.sum((ln_ret_series ** 2)))

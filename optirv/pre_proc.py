
# Module for pre-processing functions

import numpy as np

def compute_WAP(book_df):
    """
    """
    bidP1, askP1, bidQ1, askQ1 = [book_df[c] for c in ["bid_price1", "ask_price1", 
                                                       "bid_size1", "ask_size1"]]
    book_df.loc[:, "WAP"] = (bidP1*askQ1 + askP1*bidQ1)/(bidQ1 + askQ1)
    return

def compute_lnret(data, varname="WAP", group_cols=["stock_id", "time_id"]):
    """
    """
    if len(group_cols) > 0:
        data.loc[: , varname+"_lnret"] = data.groupby(group_cols)[varname].transform(
            lambda x: np.log(x / x.shift(1))
        )
    else:
        data.loc[: , varname+"_lnret"] = np.log((data[varname]/data[varname].shift(1)))
    return

def realized_vol(ln_ret_series):
    """
    """
    return np.sqrt(np.sum((ln_ret_series ** 2)))

def add_real_vol_cols(tgt_df, book_df,
                      varname="WAP_lnret", 
                      group_cols=["stock_id", "time_id"],
                      intervals = None):
    """
    """
    
    if intervals is None:
        intervals = [(0, 600)]

    for i in intervals:
        rvol = book_df.query("@i[0] <= seconds_in_bucket < @i[1]").groupby(
            group_cols, observed=True)[varname].apply(realized_vol)        
        tgt_df = tgt_df.join(rvol.rename("rvol_%d_%d"%i), on=group_cols)
        
    return tgt_df
        
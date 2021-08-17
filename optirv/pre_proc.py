
# Module for pre-processing functions

import numpy as np
import pandas as pd

def merge_book_trade(book_df, trade_df, full_frame=True, impute=True,
                     merge_cols=["stock_id", "time_id", "sec"]):
    """
    """
    
    # create using full frame if desired
    if full_frame:
        frame_df = book_df[["stock_id", "time_id"]].drop_duplicates(
            ).reset_index(drop=True)
        frame_df = frame_df.merge(pd.DataFrame(range(600), columns=["sec"]), 
                                  how="cross")
        df = frame_df.merge(book_df, on=merge_cols, how="left")
        if impute:
            df = df.fillna(method="ffill")
    else:
        df = book_df.copy()
        
    # merge with trade data
    df = df.merge(trade_df, on=merge_cols, how="left")
    if impute:
        # fwdfill missing trade prices then backfill if missing at start
        df.loc[:, "price"] = df.groupby(["stock_id", "time_id"]
                                        )["price"].ffill().bfill()
        # set price to 1 in any intervals with no trades (just in case)
        df.loc[:, "price"] = df["price"].fillna(1)
        # set order size and counts to 0 when there are no trades
        df = df.fillna(0)
        
    return df

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
        
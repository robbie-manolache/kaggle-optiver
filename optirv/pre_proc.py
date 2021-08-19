
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

def compute_WAP(df):
    """
    """
    bidP1, askP1, bidQ1, askQ1 = [df[c] for c in ["bid_price1", "ask_price1", 
                                                  "bid_size1", "ask_size1"]]
    df.loc[:, "WAP"] = (bidP1*askQ1 + askP1*bidQ1)/(bidQ1 + askQ1)
    return

def compute_lnret(df, varnames=["WAP"], group_cols=["stock_id", "time_id"]):
    """
    """
    if len(group_cols) > 0:
        for v in varnames:
            df.loc[: , v+"_lnret"] = df.groupby(group_cols)[v].transform(
                lambda x: np.log(x / x.shift(1))
            )
    else:
        for v in varnames:
            df.loc[: , v+"_lnret"] = np.log((df[v]/df[v].shift(1)))
    return

def gen_segments(df, n=3, type="obs", group_cols=["stock_id", "time_id"]):
    """
    type: "obs" (observation-based) or "sec" (time-based)
    """
    if type == "obs":
        grps = df.groupby(group_cols, observed=True)
        pctile = grps.cumcount() / grps["sec"].transform("count")
        df.loc[:, "segment"] = pd.cut(pctile, 
                                      bins=np.linspace(0, 1, n+1), 
                                      labels=range(n), 
                                      include_lowest=True)
    elif type == "sec":
        df.loc[:, "segment"] = pd.cut(df["sec"], 
                                      bins=np.linspace(0, 600, n+1), 
                                      labels=range(n), 
                                      include_lowest=True)

def realized_vol(ln_ret_series):
    """
    """
    return np.sqrt(np.sum((ln_ret_series ** 2)))
        
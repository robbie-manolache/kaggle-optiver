
# Module for pre-processing functions

import os
import numpy as np
from datetime import datetime
import pandas as pd

def merge_book_trade(book_df, trade_df, full_frame=True, impute_book=True,
                     impute_trade=False, merge_cols=["stock_id", "time_id", "sec"]):
    """
    """
    
    # create using full frame if desired
    if full_frame:
        frame_df = book_df[["stock_id", "time_id"]].drop_duplicates(
            ).reset_index(drop=True)
        frame_df = frame_df.merge(pd.DataFrame(range(600), columns=["sec"]), 
                                  how="cross")
        df = frame_df.merge(book_df, on=merge_cols, how="left")
        if impute_book:
            df = df.fillna(method="ffill")
            if "segment" in df.columns.tolist():
                df.loc[:, "segment"] = df["segment"].astype(int)
    else:
        df = book_df.copy()
        
    # merge with trade data
    df = df.merge(trade_df, on=merge_cols, how="left")

    # forward/backward fill price
    if impute_trade == False:
        # trades are to be compared to OB in the prior second
        df.loc[:, "price_f1"] = df["price"].shift(-1)
        df.loc[:, "size_f1"] = df["size"].shift(-1)
        df.loc[:, "size_l1"] = df["size"].shift(1)

        for size_col in ["size", "size_l1", "size_f1"]:
            df.loc[:, size_col] = df[size_col].fillna(0)

    else:
        # fwdfill missing trade prices then backfill if missing at start
        df.loc[:, "price"] = df.groupby(["stock_id", "time_id"]
                                        )["price"].ffill().bfill()
        # set price to 1 in any intervals with no trades (just in case)
        df.loc[:, "price"] = df["price"].fillna(1)
        # set order size and counts to 0 when there are no trades
        df = df.fillna(0)
            
    return df


def gen_merged_book_trade_var(df):
    """
    df: merged book and trade
    generate variables from merged data between trade and book data
    """
    bidP1, askP1, bidQ1, askQ1 = [df[c] for c in ["bid_price1", "ask_price1", 
                                                  "bid_size1", "ask_size1"]]
    bidP2, askP2, bidQ2, askQ2 = [df[c] for c in ["bid_price2", "ask_price2", 
                                                  "bid_size2", "ask_size2"]]
    price, size, price_f1, size_f1 = [df[c] for c in ["price", "size", "price_f1", "size_f1"]]

    df.loc[:, "m1"] = (bidP1 + askP1)/2

    sec0 = (df["sec"] == 0)

    df.loc[sec0, "ratio_size_depth1"] = size/(bidQ1 + askQ1)
    df.loc[sec0, "ratio_size_depth2"] = size/(bidQ1 + askQ1 + bidQ2 + askQ2)

    df.loc[~sec0, "ratio_size_depth1"] = size_f1/(bidQ1 + askQ1)
    df.loc[~sec0, "ratio_size_depth2"] = size_f1/(bidQ1 + askQ1 + bidQ2 + askQ2)

    return

def gen_outliers_threshold(trade_df,
                     dist_unit = ["stock_id"],
                     var_names=["size"],
                     percentile_spec=[95, 99],
                     output_dir=None):
    """
    
    """
    df = trade_df[["stock_id"]].drop_duplicates()
    
    for v in var_names:
        for i in percentile_spec:
            pct_temp = trade_df.groupby(dist_unit, observed=True)[v].apply(
                       lambda x: np.percentile(x.dropna(),i)).rename(v + "_pct_%d"%i)
            df = df.join(pct_temp, on=dist_unit)

    if output_dir is not None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(os.path.join(output_dir, "outlier_thresholds_%s.csv"%now), 
                  index=False)

    return df

def gen_outlier_flags(df, outliers_df,
                      varnames = ["WAP1_lnret", "size", "order_count"],
                      dist_unit = ["stock_id"],
                      percentile_spec=[95, 99]):

    """
    df is merged book trade, after gen_merged_book_trade_var
    outliers_df is generated from gen_outliers_threshold
    percentile_spec has to be the same as that used in gen_outliers_threshold
    """
    bidP1, askP1, bidQ1, askQ1 = [df[c] for c in ["bid_price1", "ask_price1", 
                                                  "bid_size1", "ask_size1"]]
    
    df = df.merge(outliers_df, on = dist_unit)
    
    df.loc[:, "size_add"] = df["size"] + df["size_f1"] + df["size_l1"]

    df.loc[:, "WAP1"] = (bidP1*askQ1 + askP1*bidQ1)/(bidQ1 + askQ1)
    ln_ret = np.log(df["WAP1"]/df["WAP1"].shift(1))
    df.loc[:, "WAP1_lnret"] =  ln_ret 
    df.loc[df["sec"]==0, "WAP1_lnret"] = np.nan       

    for i in percentile_spec:

        threshold = "size" + "_pct_%d"%i
        outlier_flag_name = "outlier_flag_%d"%i
        
        outliers = (df["size_add"] >= df[threshold])
        df.loc[:, outlier_flag_name] = outliers

        for v in varnames:
            adj_var_name = v + "_adj_%d"%i
            df.loc[:, adj_var_name] = df[v]            
            df.loc[outliers, adj_var_name] = 0    

    return df

def compute_WAP(df, group_cols = ["stock_id", "time_id"]):
    """
    df: book_df
    prepping OB data
    creating time length between obs
    to do: create WAP based on both levels
    """
    bidP1, askP1, bidQ1, askQ1 = [df[c] for c in ["bid_price1", "ask_price1", 
                                                  "bid_size1", "ask_size1"]]
    bidP2, askP2, bidQ2, askQ2 = [df[c] for c in ["bid_price2", "ask_price2", 
                                                  "bid_size2", "ask_size2"]]                                            
    df.loc[:, "WAP1"] = (bidP1*askQ1 + askP1*bidQ1)/(bidQ1 + askQ1)
    df.loc[:, "WAP2"] = (bidP2*askQ2 + askP2*bidQ2)/(bidQ2 + askQ2)
    df.loc[:, "midquote1"] = (bidP1 + askP1)/2
    df.loc[:, "midquote2"] = (bidP2 + askP2)/2

    df.loc[:, "max_sec"] = df.groupby(group_cols, 
                                      observed=True)["sec"].transform("max")
    df.loc[:, "sec_f1"] = df["sec"].shift(-1)
    df.loc[df["sec"]==df["max_sec"], "sec_f1"] = 600
    
    df.loc[:, "time_length"] = df["sec_f1"] - df["sec"]
    df.drop(columns=["max_sec", "sec_f1"], inplace=True)

    return

def gen_ob_slope(df):
    """
    df: book_df
    generating the slope from the LOB
    """

    cols = ["bid_price", "ask_price", "bid_size", "ask_size", "midquote"]
    bidP1, askP1, bidQ1, askQ1, m1 = [df[c+"1"] for c in cols]
    bidP2, askP2, bidQ2, askQ2, m2 = [df[c+"2"] for c in cols]
    bidQ1, askQ1, bidQ2, askQ2 = [np.log(v+1) for v in [bidQ1, askQ1, bidQ2, askQ2]]

    df.loc[:, "slope_ask"] = ((askQ2/askQ1 - 1) / ((askP2/askP1 - 1) * 100))
    df.loc[:, "slope_bid"] = ((bidQ2/bidQ1 - 1) / (np.abs(bidP2/bidP1 - 1) * 100))

    return

def gen_ob_var(df, group_cols = ["stock_id", "time_id"]):  
    """
    df: book_df
    """

    cols = ["bid_price", "ask_price", "bid_size", "ask_size", "midquote"]
    bidP1, askP1, bidQ1, askQ1, m1 = [df[c+"1"] for c in cols]
    bidP2, askP2, bidQ2, askQ2, m2 = [df[c+"2"] for c in cols]

    df.loc[:, "depth_1"] = askQ1 + bidQ1
    df.loc[:, "depth_2"] = askQ2 + bidQ2
    df.loc[:, "depth_total"] =  df["depth_1"] + df["depth_2"]
    df.loc[:, "ratio_depth1_2"] = (askQ1 + bidQ1)/(askQ2 + bidQ2)
    df.loc[:, "depth_imb1"] = (askQ1 - bidQ1)/(askQ1 + bidQ1)
    df.loc[:, "depth_imb_total"] = ((askQ1 + askQ2) - (bidQ1 + bidQ2))/df["depth_total"]

    df.loc[:, "q_spread1"] = 100 * (askP1 - bidP1)/m1
    df.loc[:, "q_spread2"] = 100 * (askP2 - bidP2)/m1
    #df.loc[:, "q_spread1_d"] = 100 * (askP1 - bidP1)
    #df.loc[:, "q_spread2_d"] = 100 * (askP2 - bidP2)
    df.loc[:, "ask_spread"] = 100 * (askP2 - askP1)/m1
    df.loc[:, "bid_spread"] = 100 * (bidP1 - bidP2)/m1

    df.loc[:, "midquote1_diff"] = (df.groupby(["stock_id", "time_id"])["bid_price1"].diff() != 0) \
                                | (df.groupby(["stock_id", "time_id"])["ask_price1"].diff() != 0)

    df.loc[:, "min_sec"] = df.groupby(group_cols, 
                           observed=True)["sec"].transform("min")                            
    df.loc[df["sec"]==df["min_sec"], "midquote1_diff"] = False
    df.drop(columns=["min_sec"], inplace=True)

    return

def compute_lnret(df, varnames=["WAP1", "WAP2"],
                  group_cols = ["stock_id", "time_id"]):
    """
    df = book_df
    power:      list of powers used to further transform log returns 
    absolute:   list of powers for which to take absolute value 
                after transforming log returns by that power
    
    NOTE: power and absolute can be different!
    """

    for v in varnames:
        
        # derive variable name
        name = v+"_lnret"
        
        # compute log returns  
        lnret = np.log((df[v]/df[v].shift(1)))  
        df.loc[:, name] = lnret 
        df.loc[:, "min_sec"] = df.groupby(group_cols, 
                               observed=True)["sec"].transform("min")
        df.loc[df["sec"]==df["min_sec"], name] = np.nan
        df.drop(columns=["min_sec"], inplace=True)

    return

def gen_trade_var(df, group_cols=["stock_id", "time_id"]):
    """
    df: trade_df
    """
    
    df.loc[:, "max_sec"] = df.groupby(group_cols, 
                                      observed=True)["sec"].transform("max")
    df.loc[:, "sec_f1"] = df["sec"].shift(-1)
    df.loc[df["sec"]==df["max_sec"], "sec_f1"] = 600

    df.loc[:, "time_length"] = df["sec_f1"] - df["sec"]
    df.drop(columns=["max_sec", "sec_f1"], inplace=True)
    
    df.loc[:, "trade_size"] = df["size"]/df["order_count"]

    return

def gen_segments_by_time(df, n=3, group_cols=["stock_id", "time_id"],
                         int_cols=["sec", "bid_size1", "ask_size1",
                                          "bid_size2", "ask_size2"],
                         return_full=True, fill_na="book"):
    """
    """
    df.loc[:, "segment"] = pd.cut(df["sec"], 
                                  bins=np.linspace(0, 600, n+1), 
                                  labels=range(n), 
                                  include_lowest=True)   
    if return_full:
        
        full_df = df[group_cols].drop_duplicates()
        full_df = full_df.merge(pd.DataFrame(range(n), columns=["segment"]), 
                                how="cross")
        new_group_cols = group_cols + ["segment"]
        full_df = full_df.merge(df, on=new_group_cols, how="left")
        
        if fill_na == "book":
            full_df = full_df.fillna(method="ffill")
        elif fill_na == "trade":
            nas = full_df["sec"].isna()
            full_df.loc[nas, "sec"] = 600 * (full_df.loc[nas, "segment"]+1) / n
            full_df.loc[:, "price"] = full_df.groupby(["stock_id", "time_id"]
                                                      )["price"].ffill().bfill()
            full_df = full_df.fillna(0)
        else:
            pass
            
        for ic in int_cols:
            if ic in full_df.columns.tolist():
                full_df.loc[:, ic] = full_df[ic].astype(int)
                
        return full_df    

# ____ Depracated ____ #             

def gen_segment_weights(df, n=3, group_cols=["stock_id", "time_id", "segment"]):
    
    seg_df = df[group_cols].drop_duplicates()
    seg_grps = df.groupby(group_cols, observed=True)

    seg_df.loc[:, "end_sec"] = seg_grps["sec"].transform("max")
    seg_df.loc[seg_df["segment"]==(n-1), "end_sec"] = 599
    seg_df.loc[:, "start_sec"] = seg_df.groupby(
        group_cols[:-1])["end_sec"].shift(1).fillna(-1)
    seg_df.loc[:, "weight"] = (seg_df["end_sec"] - seg_df["start_sec"]) / 600
          
    return seg_df[group_cols + ["weight"]]
        
def gen_segments_by_obs(df, n=3, group_cols=["stock_id", "time_id"],
                        return_segment_weights=True):
    """
    """
    grps = df.groupby(group_cols, observed=True)
    pctile = grps.cumcount() / grps["sec"].transform("count")
    df.loc[:, "segment"] = pd.cut(pctile, 
                                    bins=np.linspace(0, 1, n+1), 
                                    labels=range(n), 
                                    include_lowest=True)        
    if return_segment_weights:
        new_group_cols = group_cols + ["segment"]
        return gen_segment_weights(df, n, new_group_cols)


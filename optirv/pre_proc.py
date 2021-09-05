
# Module for pre-processing functions

import numpy as np
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
    else:
        df = book_df.copy()
        
    # merge with trade data
    df = df.merge(trade_df, on=merge_cols, how="left")

    # forward/backward fill price
    if impute_trade == False:
        # trades are to be compared to OB in the prior second
        df.loc[:, "price_f1"] = df["price"].shift(-1)
        df.loc[:, "size_f1"] = df["size"].shift(-1)

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

    df.loc[:, "time_length"] = df.groupby(group_cols, observed = True)["sec"].transform(lambda x: (x.shift(-1).fillna(600) - x))

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

def gen_ob_var(df):  
    """
    df: book_df
    """

    cols = ["bid_price", "ask_price", "bid_size", "ask_size", "midquote"]
    bidP1, askP1, bidQ1, askQ1, m1 = [df[c+"1"] for c in cols]
    bidP2, askP2, bidQ2, askQ2, m2 = [df[c+"2"] for c in cols]

    df.loc[:, "ln_depth_total"] = np.log(askQ1*askP1 + bidQ1*bidP1 + askQ2*askP2 + bidQ2*bidP2)
    df.loc[:, "ratio_depth1_2"] = (askQ1 + bidQ1)/(askQ2 + bidQ2)
    df.loc[:, "ratio_a_bdepth1"] = askQ1/bidQ1
    df.loc[:, "ratio_a_bdepth2"] = (askQ1 + askQ2)/(bidQ1 + bidQ2)

    df.loc[:, "q_spread1"] = 100 * (askP1 - bidP1)/m1
    df.loc[:, "q_spread2"] = 100 * (askP2 - bidP2)/m1
    df.loc[:, "q_spread1_d"] = 100 * (askP1 - bidP1)
    df.loc[:, "q_spread2_d"] = 100 * (askP2 - bidP2)
    df.loc[:, "ratio_askP"] = askP2/askP1
    df.loc[:, "ratio_bidP"] = bidP1/bidP2

    return

def compute_lnret(df, varnames=["WAP1", "WAP2"], 
                  group_cols=["stock_id", "time_id"],
                  power=[1], absolute=[]):
    """
    power:      list of powers used to further transform log returns 
    absolute:   list of powers for which to take absolute value 
                after transforming log returns by that power
    
    NOTE: power and absolute can be different!
    """
    
    for p in set(power + absolute):
        for v in varnames:
            
            # derive variable name
            name = v+"_lnret"
            if p == 2:
                name += "_sq"
            elif p > 2:
                name += "_pwr%d"%p
            
            # compute log returns    
            if len(group_cols) > 0:
                lnret = df.groupby(group_cols)[v].transform(
                    lambda x: np.log(x / x.shift(1)))
            else:
                lnret = np.log((df[v]/df[v].shift(1)))
            
            if p in power:
                df.loc[:, name] = lnret ** p
                
            if p in absolute:
                name += "_abs"
                lnret = np.abs(lnret ** p)
                df.loc[:, name] = lnret     
         
    return

def gen_trade_var(df, group_cols = ["stock_id", "time_id"]):
    """
    df: trade_df
    """
    df.loc[:, "time_length"] = df.groupby(group_cols, observed = True)["sec"].transform(lambda x: (x.shift(-1).fillna(600) - x))
    df.loc[:, "trade_size"] = df["size"]/df["order_count"]

    return

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

def gen_segments_by_time(df, n=3, group_cols=["stock_id", "time_id"],
                         int_cols=["sec", "bid_size1", "ask_size1",
                                          "bid_size2", "ask_size2"],
                         return_full=True):
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
        full_df = full_df.fillna(method="ffill")
        for ic in int_cols:
            if ic in full_df.columns.tolist():
                full_df.loc[:, ic] = full_df[ic].astype(int)
        return full_df
             
def gen_distribution_stats(dist_base, df, 
                            var_names=["ln_depth_total_last", "ratio_depth1_2_last",
                                    "ratio_a_bdepth1_last", "ratio_a_bdepth2_last",
                                    "q_spread1_last", "q_spread2_last",
                                    "ratio_askP_last", "ratio_bidP_last",
                                    "trade_size_med", "time_length_med",
                                    "ratio_size_depth1_ew", "ratio_size_depth2_ew"],
                            dist_unit=["stock_id"],
                            percentile_spec=[50]):
    """
    df = aggregate at stock_id-time_id
    returning distribution characteristics of a variable, default to by stock

    """
    
    for v in var_names:
        mean_name = v + "_mean"
        std_dev_name = v + "_std"
        
        mean = df.groupby(dist_unit, observed=True)[v].apply(lambda x: x.mean()).rename(mean_name)
        std_dev = df.groupby(dist_unit, observed=True)[v].apply(lambda x: x.std()).rename(std_dev_name)

        dist_base = dist_base.join(mean, on=dist_unit)
        dist_base = dist_base.join(std_dev, on=dist_unit)

        for i in percentile_spec:
            pct_temp = df.groupby(dist_unit, observed=True)[v].apply(lambda x: np.percentile(x.dropna(),i)).rename(v + "pct_%d"%i)
            dist_base = dist_base.join(pct_temp, on=dist_unit)

    return dist_base



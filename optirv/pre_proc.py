
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

def compute_WAP(df, group_cols = ["stock_id", "time_id"]):
    """
    prepping OB data
    creating time length between obs
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
    
    """

    cols = ["bid_price", "ask_price", "bid_size", "ask_size", "midquote"]
    bidP1, askP1, bidQ1, askQ1, m1 = [df[c+"1"] for c in cols]
    bidP2, askP2, bidQ2, askQ2, m2 = [df[c+"2"] for c in cols]

    df.loc[:, "ln_depth_total"] = np.log(askQ1*askP1 + bidQ1*bidP1 + askQ2*askP2 + bidQ2*bidP2)
    df.loc[:, "ratio_depth_bid1"] = (askQ1*askP1 + bidQ1*bidP1)/bidQ1*bidP1
    df.loc[:, "ratio_depth1_2"] = (askQ1*askP1 + bidQ1*bidP1)/(askQ2*askP2 + bidQ2*bidP2)
    df.loc[:, "ratio_depth_bid1_2"] = bidQ1*bidP1/bidQ2*bidP2
    df.loc[:, "ratio_depth_ask1_2"] = askQ1*askP1/askQ2*askP2

    df.loc[:, "quoted_spread1"] = 100 * (askP1 - bidP1)/m1
    df.loc[:, "quoted_spread2"] = 100 * (askP2 - bidP2)/m1
    df.loc[:, "ratio_askP"] = askP2/askP1
    df.loc[:, "ratio_bidP"] = bidP1/bidP2

    return

def gen_tweighted_var(base, df, var_name, 
                     group_cols = ["stock_id", "time_id"], 
                     weight_var = "time_length"):
    """
    generating aggregated variable weighted by time_length
    """
    weighted_var_name = var_name + "_tw"
    weighted_var = df.groupby(group_cols, observed = True)[[var_name, weight_var]].\
        apply(lambda x: np.sum(x[var_name] * x[weight_var])/np.sum(x[weight_var])).rename(weighted_var_name)
    
    base = base.join(weighted_var, on = group_cols)

    return base

def compute_lnret(df, varnames=["WAP1"], group_cols=["stock_id", "time_id"],
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

def gen_segment_weights(df, n=3, seg_type="obs", 
                        group_cols=["stock_id", "time_id", "segment"]):
    
    seg_df = df[group_cols].drop_duplicates()
    seg_grps = df.groupby(group_cols, observed=True)
    
    if seg_type == "obs":
        seg_df.loc[:, "end_sec"] = seg_grps["sec"].transform("max")
        seg_df.loc[seg_df["segment"]==(n-1), "end_sec"] = 599
        seg_df.loc[:, "start_sec"] = seg_df.groupby(
            group_cols[:-1])["end_sec"].shift(1).fillna(-1)
        seg_df.loc[:, "weight"] = (seg_df["end_sec"] - seg_df["start_sec"]) / 600
        
    elif seg_type == "sec":
        seg_df.loc[:, "N_seg"] = seg_grps["sec"].transform("count")
        seg_df.loc[:, "N_grp"] = seg_df.groupby(group_cols[:-1]).transform("sum")
        seg_df.loc[:, "weight"] = seg_df["N_seg"] / seg_df["N_grp"]
        
    return seg_df[group_cols + ["weight"]]
        
def gen_segments(df, n=3, seg_type="obs", 
                 group_cols=["stock_id", "time_id"],
                 return_segment_weights=True):
    """
    seg_type: "obs" (observation-based) or "sec" (time-based)
    """
    if seg_type == "obs":
        grps = df.groupby(group_cols, observed=True)
        pctile = grps.cumcount() / grps["sec"].transform("count")
        df.loc[:, "segment"] = pd.cut(pctile, 
                                      bins=np.linspace(0, 1, n+1), 
                                      labels=range(n), 
                                      include_lowest=True)
    elif seg_type == "sec":
        df.loc[:, "segment"] = pd.cut(df["sec"], 
                                      bins=np.linspace(0, 600, n+1), 
                                      labels=range(n), 
                                      include_lowest=True)
        
    if return_segment_weights:
        new_group_cols = group_cols + ["segment"]
        return gen_segment_weights(df, n, seg_type, new_group_cols)
        
def gen_distribution_stats(base, df, 
                            varnames=[], 
                            dist_unit=["stock_id"],
                            percentile_spec=[1, 99]):
    """
    returning distribution characteristics of a variable, default to by stock

    """
    
    for v in varnames:
        mean_name = v + "_mean"
        std_dev_name = v + "_std"
        
        mean = df.groupby(dist_unit, observed=True)[v].transform("mean").rename(mean_name)
        std_dev = df.groupby(dist_unit, observed=True)[v].transform("std").rename(std_dev_name)

        base = base.join(mean, on=dist_unit)
        base = base.join(std_dev, on=dist_unit)

        for i in percentile_spec:
            pct_temp = df.groupby(dist_unit, observed=True)[v].apply(lambda x: np.percentile(x.dropna(),i)).rename("pct_%d"%i)
            base = base.join(pct_temp, on=dist_unit)

        return base



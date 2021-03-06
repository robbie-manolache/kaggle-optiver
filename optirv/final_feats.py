
# -=-=-=-=-=-=-=-=-=-= # -=-=-=-=-=-=-=-=-=-= #
# Feature finalization # -=-=-=-=-=-=-=-=-=-= #
# -=-=-=-=-=-=-=-=-=-= # -=-=-=-=-=-=-=-=-=-= #

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd

def __ratio__(df, v, d, n, log, epsi):
    """
    """    
    df.loc[:, n] = ((df[v]+epsi) / (df[d]+epsi)) 
    if log:
        df.loc[:, n] = np.log(df[n]+epsi)

def adjust_vars(df, var_names, new_names=None, 
                pwr=1, mult=1, add=[0, 0]):
    """
    """
    if new_names is None:
        new_names = var_names
    for v, n in zip(var_names, new_names):  
        df.loc[:, n] = ((df[v]+add[0]) ** pwr)*mult + add[1]

def cap_vars(df, var_names, caps, new_names=None, add_flag=False):
    """
    """
    if new_names is None:
        new_names = var_names
    for v, c, n in zip(var_names, range(len(caps)), new_names):
        if add_flag:
            df.loc[:, n+"_capped"] = (df[v] > caps[c]).astype(int)
        df.loc[:, n] = np.where(df[v] > caps[c], caps[c], df[v])
        
def log_norm(df, var_names, new_names=None,
             add=0, above1=False, add_flag=False, mult=1):
    """
    """
    if new_names is None:
        new_names = var_names
    for v, n in zip(var_names, new_names):  
        if above1:
            df.loc[:, n] = np.where(df[v] > 1, np.log(df[v]+1e-8)+1, df[v])
            if add_flag:
                df.loc[:, n+"_abv1"] = (df[v] > 1).astype(int)
        else:
            df.loc[:, n] = np.log(df[v] + add) * mult
            
def abs_norm(df, var_names, new_names=None):
    """
    """
    if new_names is None:
        new_names = var_names
    for v, n in zip(var_names, new_names):  
        df.loc[:, n] = np.abs(df[v])
        
def interact_vars(df, vars1, vars2, new_names, op="mult"):
    """
    """
    for v1, v2, n in zip(vars1, vars2, new_names):
        if op == "mult":  
            df.loc[:, n] = df[v1] * df[v2]
        elif op == "add":
            df.loc[:, n] = df[v1] + df[v2]
        elif op == "minus":
            df.loc[:, n] = df[v1] - df[v2]
            
def compute_ratio(df, numer_vars, denom_vars, 
                  new_names=None, log=False, epsi=1e-8):
    """
    """
    if new_names is None:
        new_names = numer_vars
        
    if type(denom_vars) is str:
        
        d = denom_vars
        for v, n in zip(numer_vars, new_names):           
            __ratio__(df, v, d, n, log, epsi)
                
    elif type(denom_vars) is list:
        
        for v, d, n in zip(numer_vars, denom_vars, new_names):
            __ratio__(df, v, d, n, log, epsi)

def gen_rv_outliers(base,
                    dist_unit = ["stock_id"],
                    var_names=["WAP1_lnret_vol_all"],
                    percentile_spec=[2],
                    output_dir=None):
    """
    
    """
    df = base[["stock_id"]].drop_duplicates()
    
    for v in var_names:
        for i in percentile_spec:
            pct_temp = base.groupby(dist_unit, observed=True)[v].apply(
                       lambda x: np.percentile(x.dropna(),i)).rename(v + "_pct_%d"%i)
            df = df.join(pct_temp, on=dist_unit)

    if output_dir is not None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(os.path.join(output_dir, "rv_outlier_thresholds_%s.csv"%now, index=False))

    return df

def gen_rv_outliers_flag(base, df,
                        dist_unit=["stock_id"],
                        var_names=["WAP1_lnret_vol_all"],
                        percentile_spec=[2]):
    """
    df is from gen_rv_outliers
    percentile_spec has to be the same as in gen_rv_outliers
    """

    base = base.merge(df, on = dist_unit)

    for v in var_names:
        for i in percentile_spec:
            threshold_var = v + "_pct_%d"%i
            outliers_flag_name = "rv_flag" + "_pct_%d"%i
            outliers = (base[v] <= base[threshold_var])
            base.loc[outliers, outliers_flag_name] = 1
            base.loc[~outliers, outliers_flag_name] = 0

    return base

def seg_based_feats(df, seg_df, var_names="AUTO", seg_ranges=[[3,4]]):
    """
    """
    
    if var_names == "AUTO":
        var_names = [c for c in seg_df.columns if c not in 
                     ["stock_id", "time_id", "segment"]]
        
    out_df = df.copy()
    for sr in seg_ranges:
        new_names = [v+"_seg%d.%d"%tuple(sr) for v in var_names]
        new_df = seg_df.query("@sr[0] <= segment <= @sr[1]").groupby(
            ["stock_id", "time_id"])[var_names].mean()
        new_df.columns = new_names
        out_df = out_df.join(new_df, on=["stock_id", "time_id"]) 
        
    return out_df

def seg_based_agg(df, seg_df, feat_func="std",
                  var_names="AUTO", new_names="AUTO"):
    """
    """
    
    if var_names == "AUTO":
        var_names = [c for c in seg_df.columns if c not in 
                     ["stock_id", "time_id", "segment"]]
    
    if feat_func == "std":
        new_df = seg_df.groupby(["stock_id", "time_id"])[var_names].std()
    elif feat_func == "max":
        new_df = seg_df.groupby(["stock_id", "time_id"])[var_names].max()
    elif feat_func == "min":
        new_df = seg_df.groupby(["stock_id", "time_id"])[var_names].min()
    elif feat_func == "max.min":
        max_df = seg_df.groupby(["stock_id", "time_id"])[var_names].max()
        min_df = seg_df.groupby(["stock_id", "time_id"])[var_names].min()
        new_df = max_df - min_df
    else:
        print("Not yet implemented :(")
        
    if new_names == "AUTO":
        new_names = [c + "_seg_" + feat_func for c in new_df.columns]
        
    new_df.columns = new_names
    
    return df.join(new_df, on=["stock_id", "time_id"]) 

def seg_based_change(df, seg_df, var_names="AUTO", new_names="AUTO",
                     seg_ranges=[[0,2], [2,4]], epsi=1e-8):
    """
    """
    
    if var_names == "AUTO":
        var_names = [c for c in seg_df.columns if c not in 
                     ["stock_id", "time_id", "segment"]]
    
    seg_0 = seg_df.query("%d <= segment <= %d"%tuple(seg_ranges[0])
                         ).groupby(["stock_id", "time_id"])[var_names].mean()
    seg_1 = seg_df.query("%d <= segment <= %d"%tuple(seg_ranges[1])
                         ).groupby(["stock_id", "time_id"])[var_names].mean()
    chg_df = seg_1 / (seg_0 + epsi)
    
    if new_names == "AUTO":
        suffix = "_%d.%d_to_%d.%d"%tuple(seg_ranges[0] + seg_ranges[1])
        new_names = [c + suffix for c in chg_df.columns]
        
    chg_df.columns = new_names
    
    return df.join(chg_df, on=["stock_id", "time_id"])   
            
def stock_embed_index(df, name=["embed_index"]):
    """
    """
    n_stocks = df["stock_id"].nunique()
    stock_map = dict(zip(df["stock_id"].unique(), list(range(n_stocks))))
    df.loc[:, name] = df["stock_id"].map(stock_map)

def gen_target_change(df, target_col="target", rvol_col="WAP1_lnret_vol_all",
                      name="target_chg", pwr_adj=(1, 0.5), 
                      log_change=True, reverse=False):
    """
    """
    if reverse:
        df.loc[:, name] = (df[rvol_col]**pwr_adj[1])/(df[target_col]**pwr_adj[0])
    else:
        df.loc[:, name] = (df[target_col]**pwr_adj[0])/(df[rvol_col]**pwr_adj[1])
        
    if log_change:
        df.loc[:, name] = np.log(df[name])
    else:
        df.loc[:, name] = df[name] - 1

def gen_target_class(df, in_col='target_chg', out_col='target_class',
                     splits=[-0.5, -0.05, 0.05, 0.5]):
    """
    Categorize target into different classes by range.
    """
    
    # pad bin splits
    splits = [-np.inf] + splits + [np.inf]

    # create categorical bin column
    df.loc[:, out_col] = pd.cut(df[in_col], bins=splits, 
                                labels=list(range(len(splits)-1))
                                ).astype(int) 

def gen_weights(df, method="inv_target",
                power=0.5, pwr_adj=(1, 0.5),
                target_col="target", rvol_col="WAP1_lnret_vol_all"):
    """
    method: one of "inv_target", "target_chg"
    """
    if method == "inv_target":
        df.loc[:, "weight"] = (1/df[target_col])**power
    else:
        df.loc[:, "weight"] = np.abs(((df[target_col]**pwr_adj[0]) - 
                                    (df[rvol_col]**pwr_adj[1]))/
                                    (df[target_col]**pwr_adj[0]))**power

def add_stock_vars(df, stock_df, var_names, merge_on=["stock_id"]):
    """
    """
    return df.merge(stock_df[merge_on + var_names], on=merge_on)

def drop_vars(df, var_names):
    """
    """
    df.drop(var_names, axis=1, inplace=True)

def standardize_by_stock(df, stock_df, std_only=False, var_names=None,
                         excl_vars=["stock_id", "time_id", "target", "target_std",
                                    "target_chg", "weight", "segment"],
                         df_keys=["stock_id", "time_id"]):
    """
    """
    
    # select cols to standardize
    if var_names is None:
        x_cols = [c for c in df.columns if c not in excl_vars]
    else:
        x_cols = var_names
    frame_df = df[df_keys].copy()
    
    # extract means
    if not std_only:
        mean_df = frame_df.merge(
            stock_df[["stock_id"] + [c + "_mean" for c in x_cols]], 
            on=["stock_id"]).drop(columns=df_keys)
        mean_df.columns = x_cols
    
    # extract st. dev.
    std_df = frame_df.merge(
        stock_df[["stock_id"] + [c + "_std" for c in x_cols]], 
        on=["stock_id"]).drop(columns=df_keys)
    std_df.columns = x_cols
    
    # standardize
    if std_only:
        df.loc[:, x_cols] = df[x_cols]/std_df
    else:
        df.loc[:, x_cols] = (df[x_cols]-mean_df)/std_df
    return df

def standardize_target(df, stock_df, square=True,
                       norm_var="WAP1_lnret_vol_all"):
    """
    """
    if square:
        pwr = 2
    else:
        pwr = 1
    norm_df = df[["stock_id", "time_id", "target"]].copy()
    norm_df = norm_df.merge(stock_df[["stock_id", norm_var+"_mean", 
                                      norm_var+"_std"]], on=["stock_id"])
    df.loc[:, "target_std"] = (norm_df["target"]**pwr - norm_df[norm_var+"_mean"]
                               ) / norm_df[norm_var+"_std"]
    return df

def reshape_segments(df, n, drop_cols=["stock_id", "time_id"],
                     add_extra_axis=False):
    """
    """
    x = df.drop(drop_cols, axis=1).values
    x = np.reshape(x, (int(x.shape[0]/n), n, x.shape[1]))
    if add_extra_axis:
        return x[:, :, :, np.newaxis]
    else:
        return x
         
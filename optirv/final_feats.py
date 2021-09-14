
# -=-=-=-=-=-=-=-=-=-= # -=-=-=-=-=-=-=-=-=-= #
# Feature finalization # -=-=-=-=-=-=-=-=-=-= #
# -=-=-=-=-=-=-=-=-=-= # -=-=-=-=-=-=-=-=-=-= #

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from optirv.data_pipes import __get_func__
import optirv.feat_agg as agg

def __ratio__(df, v, d, n, log, epsi):
    """
    """    
    df.loc[:, n] = ((df[v]+epsi) / (df[d]+epsi)) 
    if log:
        df.loc[:, n] = np.log(df[n]+epsi)

def square_vars(df, var_names=["target"], new_names=["target"], pwr=2):
    """
    """
    for v, n in zip(var_names, new_names):  
        df.loc[:, n] = df[v] ** pwr

def interact_vars(df, vars1, vars2, new_names):
    """
    """
    for v1, v2, n in zip(vars1, vars2, new_names):  
        df.loc[:, n] = df[v1] * df[v2]

def compute_ratio(df, numer_vars, denom_vars, new_names, 
                  log=True, epsi=1e-8):
    """
    """
    
    if type(denom_vars) is str:
        
        d = denom_vars
        for v, n in zip(numer_vars, new_names):           
            __ratio__(df, v, d, n, log, epsi)
                
    elif type(denom_vars) is list:
        
        for v, d, n in zip(numer_vars, denom_vars, new_names):
            __ratio__(df, v, d, n, log, epsi)
            
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

def gen_weights(df, power=0.1, pwr_adj=(1, 0.5),
                target_col="target", rvol_col="WAP1_lnret_vol_all"):
    """
    """
    df.loc[:, "weight"] = np.abs(((df[target_col]**pwr_adj[0]) - 
                                  (df[rvol_col]**pwr_adj[1]))/
                                 (df[target_col]**pwr_adj[0]))**power

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

def standardize(df, group_var="stock_id",
                excl_vars=["stock_id", "time_id", "target", "target_chg", "weight"],
                df_keys=["stock_id", "time_id"], 
                input_dir=None, output_dir=None):
    """
    """
    
    x_cols = [c for c in df.columns if c not in excl_vars]
    
    if input_dir is None:
        mean_df = df.groupby(group_var)[x_cols].mean()
        std_df = df.groupby(group_var)[x_cols].std()
        if (std_df == 0).sum().sum() > 0:
            std_df[std_df == 0] = 1
    else:
        mean_df = pd.read_csv(os.path.join(output_dir, "mean.csv"), 
                              index_col=group_var)
        std_df = pd.read_csv(os.path.join(output_dir, "st_dev.csv"), 
                             index_col=group_var)
    
    if output_dir is not None:
        mean_df.to_csv(os.path.join(output_dir, "mean.csv"))
        std_df.to_csv(os.path.join(output_dir, "st_dev.csv"))
        
    mean_df = df[df_keys].join(mean_df, on=group_var)
    std_df = df[df_keys].join(std_df, on=group_var)
    
    df.loc[:, x_cols] = (df[x_cols]-mean_df[x_cols])/std_df[x_cols]

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
         
def final_feature_pipe(df, pipeline=[], task="reg", output_dir=None):
    """
    """
    
    func_map= {
        "square_vars": square_vars,
        "interact_vars": interact_vars,
        "compute_ratio": compute_ratio,
        "stock_embed_index": stock_embed_index,
        "agg_by_time_id": agg.agg_by_time_id,
        "gen_target_change": gen_target_change,
        "gen_target_class": gen_target_class,
        "gen_weights": gen_weights,
        "standardize": standardize
    }
    
    # iterate through pipeline
    for pl in pipeline:
        
        # get function object to apply
        func = __get_func__(pl["func"], func_map)
        
        # set optional arguments
        if pl["args"] is None:
            args = {}
        else:
            args = pl["args"]
            
        # perform in place or assign to output object
        func(df, **args)  
        
    if output_dir is not None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(
            output_dir, "final_proc_%s_%s.json"%(task, now)), "w") as wf:
            json.dump(pipeline, wf)                    

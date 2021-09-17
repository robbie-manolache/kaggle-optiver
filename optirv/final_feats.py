
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

def seg_based_feats(df, seg_df, feat_func="std",
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
        mean_df = pd.read_csv(os.path.join(input_dir, "mean.csv"), 
                              index_col=group_var)
        std_df = pd.read_csv(os.path.join(input_dir, "st_dev.csv"), 
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
         
def final_feature_pipe(df, aux_df=None, training=True,
                       pipeline=[], task="reg", output_dir=None):
    """
    """
    
    # function mapping
    func_map= {
        "square_vars": square_vars,
        "interact_vars": interact_vars,
        "compute_ratio": compute_ratio,
        "seg_based_feats": seg_based_feats,
        "seg_based_change": seg_based_change,
        "stock_embed_index": stock_embed_index,
        "agg_by_time_id": agg.agg_by_time_id,
        "gen_target_change": gen_target_change,
        "gen_target_class": gen_target_class,
        "gen_weights": gen_weights,
        "standardize": standardize
    }
    
    # train-only functions
    train_only = ["gen_weights", "gen_target_change", "gen_target_class"]
    
    # data mapping
    data_dict = {
        "df": df.copy(),
        "aux_df": aux_df.copy()
    }
    
    # iterate through pipeline
    for pl in pipeline:
        
        # skip training-only functions in prediction mode
        if not training and pl["func"] in train_only:
            pass
        
        else:
        
            # get function object to apply
            func = __get_func__(pl["func"], func_map)
            
            # set optional arguments
            if pl["args"] is None:
                args = {}
            else:
                args = pl["args"]
                
            # perform in place or assign to output object
            if "input" in pl.keys():
                data_dict["df"] = func(*[data_dict[d] for d 
                                        in pl["input"]], **args)
            else:
                func(data_dict["df"], **args)  
            
    if output_dir is not None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(
            output_dir, "final_proc_%s_%s.json"%(task, now)), "w") as wf:
            json.dump(pipeline, wf)  
            
    return data_dict["df"]                  

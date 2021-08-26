
# ++++++++++++++++++++++++++ #
# Feature engineering module #
# ++++++++++++++++++++++++++ #

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from optirv.data_loader import DataLoader
import optirv.pre_proc as pp

def __get_func__(func_name, func_map):
    """
    """
    if type(func_name) == str:
        return func_map[func_name]
    else:
        return func_name      

def realized_vol(ln_ret_series, subset="all"):
    """
    subset:     Must be "all", "pos" or "neg"
    """
    if subset == "pos":
        ln_ret_series = ln_ret_series * (ln_ret_series > 0)
    elif subset == "neg":
        ln_ret_series = ln_ret_series * (ln_ret_series < 0)
    
    return np.sum((ln_ret_series ** 2))

def pre_retquad(ln_ret_series):
    """
    calculate returns to the power of 4
    """
    return np.sqrt(np.sum((ln_ret_series ** 4)) * len(ln_ret_series)/3)

def pre_compute_BPV(ln_ret_series):
    """
    """
    return np.sum(ln_ret_series * ln_ret_series.shift(-1))

def add_real_vol_cols(base, df, weights=None,
                      varnames=["WAP1_lnret"], 
                      group_cols=["stock_id", "time_id"],
                      subset="all",
                      interval_col="segment",
                      intervals=None):
    """
    weights:    pandas.DataFrame containing a "weight" column and the same
                group_cols and interval_col as df. For a given interval, the 
                realized volatility will be divided by the sum of weights to 
                normalize it to the length of the entire time_id so that it 
                can be comparable no matter the length of the sub-period 
                captured by the interval.
                
    NOTE:   By default, the function uses "segment" as the interval_col, which
            is generated by the gen_segments() function. When using this function, 
            set return_segment_weights=True to generate a pandas.DataFrame that 
            can be passed to the weights argument.
    """
    
    if intervals is None:
        intervals = [(df[interval_col].min(), df[interval_col].max())]

    for v in varnames:
        for i in intervals:
            
            # derive volatility var name
            if intervals is None:
                rvol_name = "%s_vol_%s"%(v, subset)
                            
            else:
                rvol_name = "%s_vol_%s_%d_%d"%(v, subset, i[0], i[1])
                
            # compute realized volatility for each sub-segment of a time_id
            rvol = df.query("@i[0] <= %s <= @i[1]"%interval_col).groupby(
                group_cols, observed=True)[v].apply(realized_vol, 
                subset=subset).rename(rvol_name)

            # if weights provided, make sure volatility is on the same time scale
            if weights is not None:
                wgt = weights.query("@i[0] <= %s <= @i[1]"%interval_col).groupby(
                    group_cols, observed=True)[["weight"]].sum()
                wgt = wgt.join(rvol, on=group_cols)
                rvol = wgt[rvol_name] / wgt["weight"]
                rvol = rvol.rename(rvol_name)
                  
            base = base.join(rvol, on=group_cols)
        
    return base


def compute_BPV_retquad(base, df, weights=None,
                        varnames=["WAP1_lnret"], 
                        group_cols=["stock_id", "time_id"],
                        interval_col="segment",
                        intervals=None):
    """
    weights:    pandas.DataFrame containing a "weight" column and the same
                group_cols and interval_col as df. For a given interval, the 
                realized volatility will be divided by the sum of weights to 
                normalize it to the length of the entire time_id so that it 
                can be comparable no matter the length of the sub-period 
                captured by the interval.
                
    NOTE:   By default, the function uses "segment" as the interval_col, which
            is generated by the gen_segments() function. When using this function, 
            set return_segment_weights=True to generate a pandas.DataFrame that 
            can be passed to the weights argument.
    """
    
    if intervals is None:
        intervals = [(df[interval_col].min(), df[interval_col].max())]

    for v in varnames:

        abs_var = v + "_abs"

        for i in intervals:
            
            # derive BPV var name
            if intervals is None:
                BPV_name = v + "_BPV"
                BPV_jump = v + "_BPV" + "_jump"
                RQ_name = v + "_RQ"
                rvol_name = v + "_vol_all"
            else:
                BPV_name = v + "_BPV_%d_%d"%(i[0], i[1])
                BPV_jump = v + "_BPV_%d_%d"%(i[0], i[1]) + "_jump"
                RQ_name = v + "_RQ_%d_%d"%(i[0], i[1])
                rvol_name = v + "_vol_all_%d_%d"%(i[0], i[1]) 

            # check rvol in base
            if rvol_name not in base.columns:
                print("%d needs to be computed"%rvol_name)
                return

            # compute BPV and ret_quad for each sub-segment of a time_id
            BPV = df.query("@i[0] <= %s <= @i[1]"%interval_col).groupby(
                group_cols, observed=True)[abs_var].apply(pre_compute_BPV).rename(BPV_name)

            RQ = df.query("@i[0] <= %s <= @i[1]"%interval_col).groupby(
                group_cols, observed=True)[v].apply(pre_retquad).rename(RQ_name)

            # if weights provided, make sure volatility is on the same time scale
            if weights is not None:
                wgt = weights.query("@i[0] <= %s <= @i[1]"%interval_col).groupby(
                    group_cols, observed=True)[["weight"]].sum()
                wgt = wgt.join(BPV, on=group_cols)
                wgt = wgt.join(RQ, on=group_cols)

                BPV = wgt[BPV_name] * (1 / wgt["weight"]) * (np.pi/2)
                BPV = BPV.rename(BPV_name)

                RQ = wgt[RQ_name] * (1 / wgt["weight"])
                RQ = RQ.rename(RQ_name)
                  
            base = base.join(BPV, on=group_cols)
            base = base.join(RQ, on=group_cols)
            base.loc[:, BPV_jump] = np.where((base[rvol_name] - base[BPV_name]) < 0, 
                                             0, (base[rvol_name] - base[BPV_name]))
        
    return base

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

def feat_eng_pipeline(data_mode="train", data_dir=None, 
                      stock_list=None, batch_size=3,
                      pipeline=[], output_dir=None):
    """
    """
    
    # function mapping dictionary
    func_map = {
        "compute_WAP": pp.compute_WAP,
        "compute_lnret": pp.compute_lnret,
        "gen_segments": pp.gen_segments,
        "add_real_vol_cols": add_real_vol_cols,
        "compute_BPV_retquad": compute_BPV_retquad
    }
    
    # set up DataLoader and empty list to collect processed data
    dl = DataLoader(data_mode=data_mode, data_dir=data_dir)
    df_list = []
    
    # stock batch iterator
    for base, book, trade in tqdm(dl.batcher(stock_list, batch_size)):
        
        # set data dict
        data_dict = {
            "book": book,
            "trade": trade,
            "base": base
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
            if pl["output"] is None:
                func(*[data_dict[d] for d in pl["input"]], **args)
            else:
                data_dict[pl["output"]] = func(*[data_dict[d] for d 
                                                 in pl["input"]], **args)        

        # append to list
        df_list.append(data_dict["base"])
    
    # compile output and save if location provided
    df = pd.concat(df_list, ignore_index=True)
    if output_dir is not None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(os.path.join(output_dir, "%s_%s.csv"%(data_mode, now)), index=False)
        with open(os.path.join(output_dir, "%s_%s.json"%(data_mode, now)), "w") as wf:
            json.dump(pipeline, wf)
        
    return df
        
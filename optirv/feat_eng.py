
# ++++++++++++++++++++++++++ #
# Feature engineering module #
# ++++++++++++++++++++++++++ #

import numpy as np
import pandas as pd
from tqdm import tqdm
from optirv.data_loader import DataLoader
import optirv.pre_proc as pp

def __get_func__(func_name, func_map):
    """
    """
    if type(func_name) == str:
        return func_map[func_name]
    else:
        return func_name      

def realized_vol(ln_ret_series, square_root=False):
    """
    NOTE:   This function does not apply square root i.e. it is the sum of 
            squared returns!
    """
    if square_root:
        return np.sqrt(np.sum((ln_ret_series ** 2)))
    else:
        return np.sum((ln_ret_series ** 2))

def add_real_vol_cols(base, df, weights=None,
                      varnames=["WAP_lnret"], 
                      group_cols=["stock_id", "time_id"],
                      interval_col = "segment",
                      intervals = None):
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
                rvol_name = v + "_vol"
            else:
                rvol_name = v + "_vol_%d_%d"%(i[0], i[1]) 

            # compute realized volatility for each sub-segment of a time_id
            rvol = df.query("@i[0] <= %s <= @i[1]"%interval_col).groupby(
                group_cols, observed=True)[v].apply(realized_vol).rename(rvol_name)

            # if weights provided, make sure volatility is on the same time scale
            if weights is not None:
                wgt = weights.query("@i[0] <= %s <= @i[1]"%interval_col).groupby(
                    group_cols, observed=True)[["weight"]].sum()
                wgt = wgt.join(rvol, on=group_cols)
                rvol = wgt[rvol_name] / wgt["weight"]
                rvol = rvol.rename(rvol_name)
                  
            base = base.join(rvol, on=group_cols)
        
    return base

def feat_eng_pipeline(data_mode="train", data_dir=None, 
                      stock_list=None, batch_size=3,
                      pipeline = []):
    """
    """
    
    # function mapping dictionary
    func_map = {
        "compute_WAP": pp.compute_WAP,
        "compute_lnret": pp.compute_lnret,
        "add_real_vol_cols": add_real_vol_cols
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
        
    return pd.concat(df_list, ignore_index=True)
        

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

def add_real_vol_cols(base, df,
                      varnames=["WAP_lnret"], 
                      group_cols=["stock_id", "time_id"],
                      intervals = None,
                      normalize = True):
    """
    """
    
    if intervals is None:
        intervals = [(0, 600)]

    for v in varnames:
        for i in intervals:
            rvol = df.query("@i[0] <= sec < @i[1]").groupby(
                group_cols, observed=True)[v].apply(pp.realized_vol)
            if normalize:
                rvol = rvol * np.sqrt(600/(i[1]-i[0]))
            if intervals is None:
                new_name = v + "_vol"
            else:
                new_name = v + "_vol_%d_%d"%i        
            base = base.join(rvol.rename(new_name), on=group_cols)
        
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
        
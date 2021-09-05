
# +++++++++++++++++++++++++ #
# Data processing pipelines #
# +++++++++++++++++++++++++ #

import os
import json
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from optirv.data_loader import DataLoader
import optirv.pre_proc as pp
import optirv.feat_eng as fe

def __get_func__(func_name, func_map):
    """
    """
    if type(func_name) == str:
        return func_map[func_name]
    else:
        return func_name      

def feat_eng_pipeline(data_mode="train", data_dir=None, 
                      stock_list=None, batch_size=3,
                      pipeline=[], output_dir=None):
    """
    """
    
    # function mapping dictionary
    func_map = {
        
        # Pre-processing
        "merge_book_trade": pp.merge_book_trade,
        "gen_merged_book_trade_var": pp.gen_merged_book_trade_var,
        "gen_trade_var": pp.gen_trade_var,
        "gen_ob_slope": pp.gen_ob_slope,
        "gen_ob_var": pp.gen_ob_var,
        "compute_WAP": pp.compute_WAP,
        "compute_lnret": pp.compute_lnret,
        "gen_segments_by_obs": pp.gen_segments_by_obs,
        "gen_segments_by_time": pp.gen_segments_by_time,
        
        # Feature engineering
        "add_real_vol_cols": fe.add_real_vol_cols,
        "compute_BPV_retquad": fe.compute_BPV_retquad,
        "gen_weighted_var": fe.gen_weighted_var,
        "gen_trade_stats": fe.gen_trade_stats,
        "gen_last_obs": fe.gen_last_obs
        
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
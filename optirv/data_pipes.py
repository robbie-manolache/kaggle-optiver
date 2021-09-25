
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
import optirv.final_feats as ff
import optirv.feat_agg as agg

def __get_func__(func_name, func_map):
    """
    """
    if type(func_name) == str:
        return func_map[func_name]
    else:
        return func_name      

def gen_seg_base(df, key_cols=["stock_id", "time_id", "segment"]):
    """
    """
    return df[key_cols].drop_duplicates()

def feat_eng_pipeline(data_mode="train", data_dir=None, 
                      stock_list=None, batch_size=3,
                      pipeline=[], outlier_thresholds=None,
                      output_dir=None):
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
        "gen_outliers_threshold": pp.gen_outliers_threshold,
        "gen_outlier_flags": pp.gen_outlier_flags,
        
        # Intermediary
        "gen_seg_base": gen_seg_base,
        
        # Feature engineering
        "add_real_vol_cols": fe.add_real_vol_cols,
        "compute_BPV_retquad": fe.compute_BPV_retquad,
        "gen_weighted_var": fe.gen_weighted_var,
        "gen_st_dev": fe.gen_st_dev,
        "gen_max_var": fe.gen_max_var,
        "gen_trade_stats": fe.gen_trade_stats,
        "gen_adj_trade_stats": fe.gen_adj_trade_stats,
        "gen_last_obs": fe.gen_last_obs,
        "gen_depth_change": fe.gen_depth_change
        
    }
    
    # set up DataLoader and empty list to collect processed data
    dl = DataLoader(data_mode=data_mode, data_dir=data_dir)
    main_df_list = []
    seg_df_list = []
    if outlier_thresholds is None:
        outliers = []
    
    # stock batch iterator
    for base, book, trade in tqdm(dl.batcher(stock_list, batch_size)):
        
        # set data dict
        data_dict = {
            "book": book,
            "trade": trade,
            "base": base,
            "base_seg": None
        }
        
        if outlier_thresholds is not None:
            data_dict["size_tsh"] = outlier_thresholds

        # iterate through pipeline
        for pl in pipeline:
            
            if data_mode == "test" and pl["func"] == "gen_outliers_threshold":
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
                if pl["output"] is None:
                    func(*[data_dict[d] for d in pl["input"]], **args)
                else:
                    data_dict[pl["output"]] = func(*[data_dict[d] for d 
                                                    in pl["input"]], **args)        

        # append to list
        main_df_list.append(data_dict["base"])
        if data_dict["base_seg"] is not None:
            seg_df_list.append(data_dict["base_seg"])
        if "size_tsh" in data_dict.keys():
            if outlier_thresholds is None:
                outliers.append(data_dict["size_tsh"])
    
    # compile output(s) 
    main_df = pd.concat(main_df_list, ignore_index=True)
    if len(seg_df_list) > 0:
        seg_df = pd.concat(seg_df_list, ignore_index=True)
    else:
        seg_df = None
    
    # save output(s) and config if location provided
    if output_dir is not None:
        
        # set timestamp and save main training data
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_df.to_parquet(os.path.join(
            output_dir, "%s_main_%s.parquet"%(data_mode, now)), index=False)
        
        # save segment-level training data if available
        if seg_df is not None:
            seg_df.to_parquet(os.path.join(
                output_dir, "%s_seg_%s.parquet"%(data_mode, now)), index=False)        
        
        # save pipeline config
        with open(os.path.join(
            output_dir, "%s_%s.json"%(data_mode, now)), "w") as wf:
            json.dump(pipeline, wf)

        # save outlier data
        if len(outliers) > 0:
            outliers = pd.concat(outliers, ignore_index=True)
            outliers.to_csv(os.path.join(
                output_dir, "rv_outliers_%s.csv"%now), index=False)
        
    return main_df, seg_df

def final_feature_pipe(df, aux_df=None, stock_df=None, outlier_df=None,
                       training=True, pipeline=[], task="reg", output_dir=None):
    """
    """
    
    # function mapping
    func_map = {
        "adjust_vars": ff.adjust_vars,
        "cap_vars": ff.cap_vars,
        "log_norm": ff.log_norm,
        "interact_vars": ff.interact_vars,
        "compute_ratio": ff.compute_ratio,
        "gen_rv_outliers": ff.gen_rv_outliers,
        "gen_rv_outliers_flag": ff.gen_rv_outliers_flag,
        "seg_based_feats": ff.seg_based_feats,
        "seg_based_agg": ff.seg_based_agg,
        "seg_based_change": ff.seg_based_change,
        "stock_embed_index": ff.stock_embed_index,
        "agg_by_time_id": agg.agg_by_time_id,
        "agg_by_stock_id": agg.agg_by_stock_id,
        "gen_distribution_stats": agg.gen_distribution_stats,
        "gen_target_change": ff.gen_target_change,
        "gen_target_class": ff.gen_target_class,
        "gen_weights": ff.gen_weights,
        "add_stock_vars": ff.add_stock_vars,
        "drop_vars": ff.drop_vars,
        "standardize_by_stock": ff.standardize_by_stock,
        "standardize_target": ff.standardize_target
    }
    
    # train-only functions
    train_only = ["gen_weights", "gen_target_change", "gen_target_class",
                  "agg_by_stock_id", "gen_distribution_stats",
                  "standardize_target", "gen_rv_outliers"]
    
    # data mapping
    data_dict = {
        "df": df.copy(),
        "aux_df": aux_df.copy()
    }
    if stock_df is not None:
        data_dict["stock"] = stock_df.copy()
    if outlier_df is not None:
        data_dict["out_rv"] = stock_df.copy()
    
    # record time
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
            if "output" not in pl.keys():
                if "input" in pl.keys():
                    data_dict["df"] = func(*[data_dict[d] for d 
                                            in pl["input"]], **args)    
                else:
                    func(data_dict["df"], **args)  
                
            # add any interim outputs and save
            else:
                if "input" in pl.keys():
                    data_dict[pl["output"]] = func(
                        *[data_dict[d] for d in pl["input"]], **args)
                else:
                    data_dict[pl["output"]] = func(data_dict["df"], **args)
                    
                # save any stock-level stats for pred deployment
                if pl["output"] == "stock" and output_dir is not None:
                    data_dict[pl["output"]].to_csv(os.path.join(
                        output_dir, "stocks_%s_%s.csv"%(task, now)
                    ), index=False)
                if pl["output"] == "out_rv" and output_dir is not None:
                    data_dict[pl["output"]].to_csv(os.path.join(
                        output_dir, "out_rv_%s_%s.csv"%(task, now)
                    ), index=False)                
            
    if output_dir is not None:
        with open(os.path.join(
            output_dir, "final_proc_%s_%s.json"%(task, now)), "w") as wf:
            json.dump(pipeline, wf)  
            
    return data_dict["df"]                  

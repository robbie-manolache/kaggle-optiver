
# -=-=-=-=-=-=-=-=-=-=-=-=- # -=-=-=-=-=-=-=-=-=-=-=-=- #
# Train RV Change Classifer # -=-=-=-=-=-=-=-=-=-=-=-=- #
# -=-=-=-=-=-=-=-=-=-=-=-=- # -=-=-=-=-=-=-=-=-=-=-=-=- #

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from optirv.eval_tools import predict_target_class
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, log_loss

def train_lgbm_classifier(config, 
                          train_df, 
                          valid_df=None, 
                          model_prefix="lgbm_class",
                          output_dir=None,
                          save_model=False,
                          time_stamp=None,
                          verbose=False):
    """
    !!! save_model=True does nothing if output_dir=None !!!
    """
    
    # Load config params
    params = config["params"] # dict
    nround = config["nround"] # int
    e_stop = config["e_stop"] # int
    x_cols = config["x_cols"] # list
    x_cats = config["x_cats"] # list / None
    target = config["target"] # str
    
    # adjust x_cats to default if None
    if x_cats is None:
        x_cats = "auto"
    
    # create lightGBM datasets
    train_lgb = lgb.Dataset(train_df[x_cols], label=train_df[target],
                            categorical_feature=x_cats)
    if valid_df is None:
        valid_lgb, e_stop = None, None
    else:
        valid_lgb = lgb.Dataset(valid_df[x_cols], label=valid_df[target],
                                categorical_feature=x_cats)
    
    # train model
    model = lgb.train(params=params, 
                      train_set=train_lgb,
                      valid_sets=valid_lgb,
                      num_boost_round=nround, 
                      early_stopping_rounds=e_stop,
                      categorical_feature=x_cats,
                      verbose_eval=verbose)
    
    # generate predictions
    if valid_df is None:
        pred_df = predict_target_class(train_df, model)
    else:
        pred_df = predict_target_class(valid_df, model)
    
    # save outputs as required
    if output_dir is not None:
                
        # create model name
        if time_stamp is None:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            now = time_stamp
        model_name =  "%s_%s"%(model_prefix, now)
        
        # save model and config
        if save_model:
            model.save_model(os.path.join(output_dir, "%s.txt"%model_name))
            with open(os.path.join(output_dir, "%s_cfg.json"%model_name), "w") as wf:
                json.dump(config, wf)
            
    return model, pred_df

def classifier_CV(df, config,
                  train_func="train_lgbm_classifier",
                  model_prefix="lgbm_class",
                  n_splits=5, split_seed=42,
                  output_dir=None):
    """
    """
    
    # pick training function
    func_dict = {
        "train_lgbm_classifier": train_lgbm_classifier
    }
    train_func = func_dict[train_func]
    
    # setup
    time_ids = df["time_id"].unique()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    all_preds = []
    
    # iterate thru the folds
    for train_idx, test_idx in kf.split(time_ids):
        
        # select data
        train_times, test_times = time_ids[train_idx], time_ids[test_idx]
        train_df = df.query("time_id in @train_times").copy()
        valid_df = df.query("time_id in @test_times").copy()
        
        # train model
        model, pred_df = train_func(config, train_df, valid_df)
        all_preds.append(pred_df)
        
    # compile all_preds
    all_preds = pd.concat(all_preds, ignore_index=True)
    
    # train full model
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    model, pred_df = train_func(config, train_df=df, valid_df=None,
                                model_prefix=model_prefix,
                                output_dir=output_dir,
                                save_model=True, time_stamp=now)
    
    if output_dir is not None:
        
        # create results file for all out-of-sample preds
        target = config["target"]
        log_loss_preds = log_loss(all_preds[target], 
                                  all_preds[[c for c in all_preds.columns 
                                             if c.startswith("class_")]])
        results = {
            "log_loss": log_loss_preds,
            "accuracy": np.exp(-log_loss_preds),
            "f1_scores": f1_score(all_preds[target], all_preds["pred_class"],
                                  average=None).tolist(),
            "f1_micro": f1_score(all_preds[target], all_preds["pred_class"],
                                 average="micro"),   
            "f1_macro": f1_score(all_preds[target], all_preds["pred_class"],
                                 average="macro"),
            "f1_weigthed": f1_score(all_preds[target], all_preds["pred_class"],
                                    average="weighted")  
        }
        
        # save preds and results
        with open(os.path.join(
            output_dir, "%s_%s_results.json"%(model_prefix, now)), "w") as wf:
            json.dump(results, wf)                                      
        all_preds.to_parquet(os.path.join(output_dir, "%s_%s_preds.parquet"%
                                          (model_prefix, now)), index=False)        
    
    return model, all_preds

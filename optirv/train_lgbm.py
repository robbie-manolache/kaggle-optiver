
# -=-=-=-=-=-=-=-=-=-=-=-=- # -=-=-=-=-=-=-=-=-=-=-=-=- #
# Train RV Change Classifer # -=-=-=-=-=-=-=-=-=-=-=-=- #
# -=-=-=-=-=-=-=-=-=-=-=-=- # -=-=-=-=-=-=-=-=-=-=-=-=- #

import os
import json
from datetime import datetime
from lightgbm.engine import train
import numpy as np
import pandas as pd
import lightgbm as lgb
from pandas.core.reshape.concat import concat
from optirv import eval_tools
from optirv.eval_tools import predict_target, predict_target_class, \
    adjust_preds, multi_log_loss, cv_reg_stats
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

def __score_classifier___(preds, target):
    """
    """
    log_loss_preds = multi_log_loss(preds, target)
    return {"log_loss": log_loss_preds,
            "accuracy": np.exp(-log_loss_preds),
            "f1_scores": f1_score(preds[target], preds["pred_class"],
                                  average=None).tolist(),
            "f1_micro": f1_score(preds[target], preds["pred_class"],
                                 average="micro"),   
            "f1_macro": f1_score(preds[target], preds["pred_class"],
                                 average="macro"),
            "f1_weighted": f1_score(preds[target], preds["pred_class"],
                                    average="weighted")  
    }

def train_lgbm_model(config, 
                     train_df, 
                     valid_df=None, 
                     model_prefix="lgbm",
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
    weight = config["weight"] # str / None
    target = config["target"] # str
    
    # adjust x_cats to default if None
    if x_cats is None:
        x_cats = "auto"
    
    # get weights if provided
    if weight is None:
        weight_vec = None
    else:
        weight_vec = train_df[weight]
    
    # create lightGBM datasets
    train_lgb = lgb.Dataset(train_df[x_cols], 
                            label=train_df[target],
                            categorical_feature=x_cats, 
                            weight=weight_vec)
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
    
    # set up eval_cols
    eval_cols = [target]
    if "target" not in eval_cols:
        eval_cols.append("target")
    
    # generate predictions
    if params["objective"] == "multiclass":
        if valid_df is None:      
            pred_df = predict_target_class(train_df, model)
        else:
            pred_df = predict_target_class(valid_df, model)
    elif params["objective"] == "rmse":
        if valid_df is None:      
            pred_df = predict_target(train_df, model, eval_cols=eval_cols)
        else:
            pred_df = predict_target(valid_df, model, eval_cols=eval_cols)
    elif params["objective"] == "quantile":
        if valid_df is None:      
            pred_df = predict_target(train_df, model, eval_cols=eval_cols, 
                                     alpha=params["alpha"])
        else:
            pred_df = predict_target(valid_df, model, eval_cols=eval_cols, 
                                     alpha=params["alpha"])
    
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

def lgbm_CV(df, config, norm_df=None,
            train_func="train_lgbm_model",
            n_splits=5, split_seed=42, 
            sqr_target=True, min_rv=0.00011,
            model_prefix="lgbm", output_dir=None):
    """
    """
    
    # pick training function
    func_dict = {
        "train_lgbm_model": train_lgbm_model
    }
    train_func = func_dict[train_func]
    
    # get target name and set up eval_cols
    target = config["target"]
    eval_cols = [target]
    if "target" not in eval_cols:
        eval_cols.append("target")
    
    # setup
    time_ids = df["time_id"].unique()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    all_preds = []
    train_stats = []
    fold_num = 0
    
    # iterate thru the folds
    for train_idx, test_idx in kf.split(time_ids):
        
        # select data
        train_times, test_times = time_ids[train_idx], time_ids[test_idx]
        train_df = df.query("time_id in @train_times").copy()
        valid_df = df.query("time_id in @test_times").copy()
        
        # train model
        model, pred_df = train_func(config, train_df, valid_df)
        
        # capture preds
        fold_num += 1
        pred_df.loc[:, "fold"] = fold_num
        all_preds.append(pred_df)
        
        # include training stats
        if config["params"]["objective"] == "rmse":
            train_cv = predict_target(train_df, model, eval_cols=eval_cols)
            train_cv.loc[:, "Fold"] = fold_num
            
            # adjust for any standardization
            if norm_df is not None:
                train_cv = adjust_preds(train_cv, norm_df, min_rv, sqr_target)                
                pred_col = "pred_adj"
            else:
                pred_col = "pred"                 
            
            # calculate metrics and append
            train_cv = cv_reg_stats(train_cv, 0, target="target",  
                                    pred_col=pred_col, mode="train-cv")
            train_cv.loc[:, "Fold"] = "Fold_%d"%fold_num
            train_stats.append(train_cv)
            print("RMSE: %.4f | MAPE: %.4f | RMSPE %.4f"%tuple(
                train_cv.loc[0, ["RMSE", "MAPE", "RMSPE"]]))
        
        print("|%s Fold %d complete! %s|"%("-"*25, fold_num, "-"*25))
        
    # compile all_preds
    all_preds = pd.concat(all_preds, ignore_index=True)
    
    # train full model
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    model, pred_df = train_func(config, train_df=df, valid_df=None,
                                model_prefix=model_prefix,
                                output_dir=output_dir,
                                save_model=True, time_stamp=now)
    
    # add folds to in-sample preds for comparison
    pred_df = pred_df.merge(all_preds[["stock_id", "time_id", "fold"]], 
                            on=["stock_id", "time_id"])
    
    # adjust for any standardization
    if norm_df is not None:
        pred_df = adjust_preds(pred_df, norm_df, min_rv, sqr_target)
        all_preds = adjust_preds(all_preds, norm_df, min_rv, sqr_target)
    
    # save outputs if not none
    if output_dir is not None:
        
        # classification eval stats
        if config["params"]["objective"] == "multiclass":
            results = {
                "training": __score_classifier___(pred_df, target),
                "validation": __score_classifier___(all_preds, target)
            }
            with open(os.path.join(
                output_dir, "%s_%s_results.json"%(model_prefix, now)), "w") as wf:
                json.dump(results, wf) 
               
        # regression eval stats         
        elif config["params"]["objective"] == "rmse":
            results = pd.concat(
                [cv_reg_stats(all_preds, n_splits, target="target",  
                              pred_col=pred_col, mode="test"),
                 pd.concat(train_stats, ignore_index=True),
                 cv_reg_stats(pred_df, n_splits, target="target",  
                              pred_col=pred_col, mode="train-all")],
                ignore_index=True)
            results.to_csv(os.path.join(
                output_dir, "%s_%s_results.csv"%(model_prefix, now)), index=False) 
        
        # quantile eval stats
        # TBC
        
        # save preds                               
        all_preds.to_parquet(os.path.join(output_dir, "%s_%s_preds.parquet"%
                                          (model_prefix, now)), index=False)        
    
    return model, all_preds

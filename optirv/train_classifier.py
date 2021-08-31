
# -=-=-=-=-=-=-=-=-=-=-=-=- # -=-=-=-=-=-=-=-=-=-=-=-=- #
# Train RV Change Classifer # -=-=-=-=-=-=-=-=-=-=-=-=- #
# -=-=-=-=-=-=-=-=-=-=-=-=- # -=-=-=-=-=-=-=-=-=-=-=-=- #

import os
import json
from datetime import datetime
import lightgbm as lgb
from optirv.eval_tools import predict_target_class

def train_lgbm_classifier(config, 
                          train_df, 
                          valid_df=None, 
                          model_prefix="lgbm_class_",
                          output_dir=None,
                          save_preds=False,
                          save_model=False,
                          verbose=False):
    
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
        
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name =  "%s_%s.txt"%(model_prefix, now)
        
        if save_model:
            model.save_model(os.path.join(output_dir, "%s.txt"%model_name))
            with open(os.path.join(output_dir, "%s_cfg.json"%model_name), "w") as wf:
                json.dump(config, wf) 
        
        if save_preds:
            pred_df.to_csv(os.path.join(output_dir, "%s.csv"%model_name),
                           index=False)
            
    return model, pred_df

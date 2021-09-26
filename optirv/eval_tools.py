
# +++++++++++++++++++++ #
# Module for evaluation #
# +++++++++++++++++++++ #

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

def rmspe_calc(y_true, y_pred):
    """
    """
    return np.sqrt(np.mean(((y_true-y_pred)/y_true) ** 2))

def cv_reg_stats(preds, n_splits, target,
                 pred_col="pred", mode="test"):
    """
    """
    results = []
    for f in range(n_splits+1):
        if f == 0:
            pred_df = preds.copy()
            fold = "All"
        else:
            pred_df = preds.query("fold == @f").copy()
            fold = "Fold_" + str(f)
        
        mse = np.mean(np.square(pred_df[target] - pred_df[pred_col]))
        mape = np.mean(np.abs(pred_df[target] - pred_df[pred_col]))
        rmspe = rmspe_calc(pred_df[target], pred_df[pred_col])       
        results.append({"Fold": fold, "Mode": mode, 
                        "RMSE": np.sqrt(mse)*100, 
                        "MAPE": mape*100, "RMSPE": rmspe})
        
    return pd.DataFrame(results)   

def predict_target(eval_df, model, quantile=None,
                   key_cols=["stock_id", "time_id"],
                   eval_cols=["target_chg"]):
    """
    """
    # copy eval_df frame
    df = eval_df.copy()[key_cols + eval_cols]

    # derive predcol name
    pred_col = "pred"
    #if quantile is not None:
    #    pred_col += "_%d"%(int(100*quantile))
    
    # insert preds
    df.loc[:, pred_col] = model.predict(eval_df[model.feature_name()])
    if "target_chg" in eval_cols:
        df.loc[:, pred_col] = np.exp(df[pred_col])*df["WAP1_lnret_vol_all"]
    return df

def adjust_preds(df, norm_df, min_rv=0.00011, sqr_target=True):
    """
    """
    df = df.merge(norm_df, on=["stock_id"], how="left")

    df.loc[:, "pred_adj"] = df["pred"] * df["std"]
    df.drop("std", axis=1)

    if "mean" in norm_df.columns:
        df.loc[:, "pred_adj"] = df["pred_adj"] + df["mean"]
        df.drop("mean", axis=1)
        
    df.loc[df["pred_adj"]<(0.5*min_rv), "pred_adj"] = min_rv
    if sqr_target:
        df.loc[:, "pred_adj"] = np.sqrt(df["pred_adj"])
        
    return df   

def predict_target_class(eval_df, model,
                         key_cols=["stock_id", "time_id"],
                         eval_cols=["target_chg", "target_class"]):
    """
    Predict target class using LGBM classifier.
    """
    
    # copy eval_df frame
    df = eval_df.copy()[eval_cols]
    
    # make predictions
    preds = model.predict(eval_df[model.feature_name()])
    
    # add predictions to df
    pred_cols = ["class_" + str(i) for i in range(preds.shape[1])]
    df.loc[:, pred_cols] = preds
    df.loc[:, "pred_class"] = preds.argmax(axis=1)
    
    return df

def multi_log_loss(pred_df, target="target_class"):
    """
    """
    pred_cols = [c for c in pred_df.columns if c.startswith("class_")]
    return log_loss(pred_df[target], pred_df[pred_cols])

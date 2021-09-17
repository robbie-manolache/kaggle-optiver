
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

def cv_reg_stats(preds, n_splits, target, ln_target, sqr_target, mode="test"):
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
        
        mse = np.mean(np.square(pred_df[target] - pred_df["pred"]))
        for c in ["pred", target]:
            if ln_target:
                pred_df.loc[:, c] = np.exp(pred_df[c])
            else:
                if target != "target":
                    pred_df.loc[:, c] = pred_df[c] + 1
            if sqr_target:
                pred_df.loc[:, c] = np.sqrt(pred_df[c])
        rmspe = rmspe_calc(pred_df[target], pred_df["pred"])
        
        results.append({"Fold": fold, "Mode": mode,
                        "RMSE": np.sqrt(mse), "MSE": mse, "RMSPE": rmspe})
        
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
    if quantile is not None:
        pred_col += "_%d"%(int(100*quantile))
    
    # insert preds
    df.loc[:, pred_col] = model.predict(eval_df[model.feature_name()])
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

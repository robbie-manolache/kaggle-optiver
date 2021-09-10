
# +++++++++++++++++++++ #
# Module for evaluation #
# +++++++++++++++++++++ #

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

def rmspe(y_true, y_pred):
    """
    """
    return np.sqrt(np.mean(((y_true-y_pred)/y_true) ** 2))

def predict_target_class(eval_df, model,
                         eval_cols=["stock_id", "time_id", 
                                    "target_chg", "target_class"]):
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

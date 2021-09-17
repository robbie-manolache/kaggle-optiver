
# -=-=-=-=-=-=-=-=-=-=-=-=- # -=-=-=-=-=-=-=-=-=-=-=-=- #
# Train RV Regression Model # -=-=-=-=-=-=-=-=-=-=-=-=- #
# -=-=-=-=-=-=-=-=-=-=-=-=- # -=-=-=-=-=-=-=-=-=-=-=-=- #

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd

import tensorflow as tf
import keras.layers as kl
import keras.regularizers as kreg
import keras.backend as K
from keras.optimizers import Adam
from sklearn.model_selection import KFold

from optirv.final_feats import reshape_segments
from optirv.eval_tools import rmspe_calc, cv_reg_stats

def rmspe_loss(y_true, y_pred):
    """
    """   
    return K.sqrt(K.mean(K.square((y_true-y_pred)/(y_true+1e-6)), axis=-1))

def __conv_layer__(x, c):
    """
    """
    conv_x = kl.Conv2D(filters=c["filters"], 
                       kernel_size=(1, c["shape"][1]),
                       activation=c["conv_acti"],
                       kernel_regularizer=kreg.l2(c["conv_reg"])
                       )(x)
    conv_x = kl.Flatten()(conv_x)
    if c["dense"] > 0:
        conv_x = kl.Dense(c["dense"],
                            activation=c["dense_acti"],
                            kernel_regularizer=kreg.l2(c["dense_reg"])
                            )(conv_x)
    return conv_x

def __lstm_layer__(x, c):
    """
    """
    lstm_x = kl.LSTM(units=c["units"],
                     kernel_regularizer=kreg.l2(c["knl_reg"]),
                     recurrent_regularizer=kreg.l2(c["rec_reg"]),
                     activity_regularizer=kreg.l2(c["act_reg"])
                     )(x)
    if c["dense"] > 0:
        lstm_x = kl.Dense(c["dense"],
                            activation=c["dense_acti"],
                            kernel_regularizer=kreg.l2(c["dense_reg"])
                            )(lstm_x)
    return lstm_x

def build_NN_model(dense_in=[],
                   conv_in=[],
                   lstm_in=[],
                   class_in=[],
                   embed={"mult": True, "const": 1, "N": None},
                   extra_layer=None,
                   out_layer={"acti": "linear", "reg": 1e-5}):
    """
    """
    
    all_inputs = []   
    n_mult = 0
    
    # if using embeddings, init the layer input
    if embed["mult"] or (embed["const"] > 0):
        emb_in = kl.Input(shape=1)
        all_inputs.append(emb_in)
    
    # iterate and accumulate any flat data inputs for dense layers
    x_out = []
    for d in dense_in:
        all_inputs.append(kl.Input(shape=d["shape"]))
        n_mult += d["N"]
        x_out.append(kl.Dense(d["N"], activation=d["acti"],
                              kernel_regularizer=kreg.l2(d["reg"])
                              )(all_inputs[-1]))
    
    # iterate and accumulate inputs for cnn layers
    for c in conv_in:
        all_inputs.append(kl.Input(shape=c["shape"]))
        if c["dense"] > 0:
            n_mult += c["dense"]
        else:
            n_mult += (c["filters"] * c["shape"][0])
        x_out.append(__conv_layer__(all_inputs[-1], c))
        
    # iterate and accumulate inputs for LSTM layers
    for l in lstm_in:
        all_inputs.append(kl.Input(shape=l["shape"]))
        if l["dense"] > 0:
            n_mult += l["dense"]
        else:
            n_mult += l["units"]
        x_out.append(__lstm_layer__(all_inputs[-1], l))
    
    # iterate and accumulate class prediction inputs
    for xc in class_in:
        all_inputs.append(kl.Input(shape=xc["N"]))
        n_mult += xc["N"]
        x_out.append(all_inputs[-1])
    
    # concatenate multiple inputs
    if len(x_out) > 1:
        x_out = kl.Concatenate()(x_out)
    else:
        x_out = x_out[0]
    
    # multiply by embedding layer weights
    if embed["mult"]:
        emb_mult = kl.Embedding(embed["N"], n_mult, 
                                input_length=1)(emb_in)
        emb_mult = kl.Flatten()(emb_mult)
        x_out = kl.Multiply()([x_out, emb_mult])
    
    # add embedding fixed effect constants    
    if embed["const"] > 0:
        emb_const = kl.Embedding(embed["N"], embed["const"], 
                                 input_length=1)(emb_in)
        emb_const = kl.Flatten()(emb_const)
        x_out = kl.Concatenate()([x_out, emb_const])
    
    # add extra layer if provided
    if extra_layer is not None:
        x_out = kl.Dense(extra_layer["N"], activation=extra_layer["acti"],
                         kernel_regularizer=kreg.l2(extra_layer["reg"]))(x_out)
    
    # final output layer    
    y_out = kl.Dense(1, activation=out_layer["acti"],
                     kernel_regularizer=kreg.l2(out_layer["reg"]))(x_out)
    
    # connect model layers and return
    model = tf.keras.Model(inputs=all_inputs, outputs=y_out)
    
    return model
        
def train_NN_model(model_config, x_train, y_train,
                   x_valid=None, pred_df=None,
                   loss="mse", weights=None, lr=0.0025,
                   epochs=[50, 25], batches=[15000, 97000],
                   shuffle=True, verbose=0,
                   output_dir=None, model_prefix="NN",
                   save_model=False, time_stamp=None):
    """
    weight:     vector of weights, same length as y_train
    """
    
    # create model object and compule
    model = build_NN_model(**model_config)
    model.compile(loss=loss,
                  optimizer=Adam(learning_rate=lr))
    
    # train model
    for n, b in zip(epochs, batches):
        model.fit(x=x_train, y=y_train, 
                  epochs=n, batch_size=b, 
                  sample_weight=weights,
                  shuffle=shuffle, 
                  verbose=verbose)
    
    # generate predictions if 
    if (x_valid is not None) and (pred_df is not None):
        y_pred = model.predict(x_valid, batch_size=10000)
        pred_df.loc[:, "pred"] = y_pred
        
    # save outputs as required
    if output_dir is not None:
        
        # create model name
        if time_stamp is None:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            now = time_stamp
        model_name =  "%s_%s"%(model_prefix, now)
        
        config = {
            "loss": loss,
            "lr": lr,
            "epochs": epochs,
            "batches": batches,
            "shuffle": shuffle
        }
        
        # save model and any relevant configs
        if save_model:
            model.save(os.path.join(output_dir, model_name))
            
            # model config
            with open(os.path.join(output_dir, "%s_model_cfg.json"%model_name), "w") as wf:
                json.dump(model_config, wf)
            
            # train config
            with open(os.path.join(output_dir, "%s_train_cfg.json"%model_name), "w") as wf:
                json.dump(config, wf)
            
    return model, pred_df

def regression_CV(main_df, seg_df, class_df,
                  data_config, model_config, train_config,
                  n_splits=5, split_seed=42, 
                  weight_col=None, target="target_chg", 
                  sqr_target=True, ln_target=True,
                  model_prefix="NN", output_dir=None):
    """
    weight_col: name of column in main_df containing weights
    """

    # extract metadata
    time_ids = main_df["time_id"].unique()
    n_stocks = main_df["stock_id"].nunique()
    n_seg = seg_df["segment"].nunique()
    
    # sort input dfs
    main_df = main_df.sort_values(["stock_id", "time_id"])
    seg_df = seg_df.sort_values(["stock_id", "time_id", "segment"])
    class_df = class_df.sort_values(["stock_id", "time_id"])
    
    # initiate KF
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    all_preds = []
    fold_num = 0
    
    # loop thru folds
    for train_idx, test_idx in kf.split(time_ids):
        
        # select data
        train_times, test_times = time_ids[train_idx], time_ids[test_idx]
        
        # split data    
        main_train = main_df.query("time_id in @train_times").copy()
        main_test = main_df.query("time_id in @test_times").copy()
        seg_train = seg_df.query("time_id in @train_times").copy()
        seg_test = seg_df.query("time_id in @test_times").copy()
        class_train = class_df.query("time_id in @train_times").copy()
        class_test = class_df.query("time_id in @test_times").copy()
        
        # reshape segment data
        seg_train = reshape_segments(seg_train, n_seg)
        seg_test = reshape_segments(seg_test, n_seg)
        
        # apply data config
        x_train, x_test = [], []
        for k, v in data_config.items():
            if k == "embed":
                x_train.append(main_train[v])
                x_test.append(main_test[v])
            elif k == "main":
                for col_set in v:
                    x_train.append(main_train[col_set])
                    x_test.append(main_test[col_set])
            elif k == "seg":
                if v:
                    x_train.append(seg_train)
                    x_test.append(seg_test)
            elif k == "class":
                x_train.append(class_train[v])
                x_test.append(class_test[v])
        
        # get weights
        if weight_col is not None:
            weights = main_train[weight_col]
        else:
            weights = None
        
        pred_frame = main_test[["stock_id", "time_id", target]].copy()        
        model, pred_df = train_NN_model(model_config, 
                                        x_train, 
                                        main_train[target],
                                        x_test, 
                                        pred_frame,
                                        weights=weights,
                                        **train_config)
        
            
        fold_num += 1
        pred_df.loc[:, "fold"] = fold_num
        all_preds.append(pred_df)
        print("|%s Fold %d complete! %s|"%("-"*25, fold_num, "-"*25))
    
    # compile all_preds
    all_preds = pd.concat(all_preds, ignore_index=True)
    
    # apply data config
    x_train = []
    for k, v in data_config.items():
        if k == "embed":
            x_train.append(main_df[v])
        elif k == "main":
            for col_set in v:
                x_train.append(main_df[col_set])
        elif k == "seg":
            if v:
                x_train.append(reshape_segments(seg_df, n_seg))
        elif k == "class":
            x_train.append(class_df[v])
    
    # train full model
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    model, pred_df = train_NN_model(model_config, 
                                    x_train,
                                    main_df[target],
                                    weights=weights,
                                    save_model=True,
                                    model_prefix=model_prefix,
                                    output_dir=output_dir,
                                    time_stamp=now,
                                    **train_config)
    
    # save outputs
    if output_dir is not None:
        
        # save preds and data config       
        all_preds.to_parquet(os.path.join(output_dir, "%s_%s_preds.parquet"%
                                          (model_prefix, now)), index=False)
        with open(os.path.join(
            output_dir, "%s_%s_data_cfg.json"%(model_prefix, now)), "w") as wf:
            json.dump(data_config, wf) 
            
        # results
        results = cv_reg_stats(all_preds, n_splits, target,
                               ln_target, sqr_target)
        results.to_csv(os.path.join(
            output_dir, "%s_%s_results.csv"%(model_prefix, now)), index=False) 
                
    return model, all_preds 

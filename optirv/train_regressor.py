
# -=-=-=-=-=-=-=-=-=-=-=-=- # -=-=-=-=-=-=-=-=-=-=-=-=- #
# Train RV Regression Model # -=-=-=-=-=-=-=-=-=-=-=-=- #
# -=-=-=-=-=-=-=-=-=-=-=-=- # -=-=-=-=-=-=-=-=-=-=-=-=- #

import os
import json
from datetime import datetime
import tensorflow as tf
import keras.layers as kl
import keras.regularizers as kreg
import keras.backend as K
from keras.optimizers import Adam

def rmspe_loss(y_true, y_pred):
    """
    """   
    return K.sqrt(K.mean(K.square((y_true-y_pred)/(y_true+1e-6)), axis=-1))

def build_NN_model(dense_in=[],
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
                   loss="mse", lr=0.0025,
                   epochs=[50, 25], batches=[15000, 97000],
                   shuffle=True, verbose=0,
                   output_dir=None, model_prefix="NN_",
                   save_model=False, save_preds=False):
    """
    """
    
    # create model object and compule
    model = build_NN_model(**model_config)
    model.compile(loss=loss,
                  optimizer=Adam(learning_rate=lr))
    
    # train model
    for n, b in zip(epochs, batches):
        model.fit(x=x_train, y=y_train, 
                  epochs=n, batch_size=b, 
                  shuffle=shuffle, 
                  verbose=verbose)
    
    # generate predictions if 
    if (x_valid is not None) and (pred_df is not None):
        y_pred = model.predict(x_valid, batch_size=10000)
        pred_df.loc[:, "pred"] = y_pred
        
    # save outputs as required
    if output_dir is not None:
        
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        
        # save predictions        
        if save_preds and (x_valid is not None) and (pred_df is not None):
            pred_df.to_csv(os.path.join(output_dir, "%s.csv"%model_name),
                           index=False)
            
    return model, pred_df

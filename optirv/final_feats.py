
# -=-=-=-=-=-=-=-=-=-= # -=-=-=-=-=-=-=-=-=-= #
# Feature finalization # -=-=-=-=-=-=-=-=-=-= #
# -=-=-=-=-=-=-=-=-=-= # -=-=-=-=-=-=-=-=-=-= #

import numpy as np
from optirv.feat_eng import __get_func__

def __ratio__(df, v, d, n, log, epsi):
    """
    """    
    df.loc[:, n] = ((df[v] + epsi) / (df[d]+epsi)) 
    if log:
        df.loc[:, n] = np.log(df[n])

def square_vars(df, var_names=["target"], new_names=["target"], pwr=2):
    """
    """
    for v, n in zip(var_names, new_names):  
        df.loc[:, n] = df[v] ** pwr

def interact_vars(df, vars1, vars2, new_names):
    """
    """
    for v1, v2, n in zip(vars1, vars2, new_names):  
        df.loc[:, n] = df[v1] * df[v2]

def compute_ratio(df, numer_vars, denom_vars, new_names, 
                  log=True, epsi=1e-8):
    """
    """
    
    if type(denom_vars) is str:
        
        d = denom_vars
        for v, n in zip(numer_vars, new_names):           
            __ratio__(df, v, d, n, log, epsi)
                
    elif type(denom_vars) is list:
        
        for v, d, n in zip(numer_vars, denom_vars, new_names):
            __ratio__(df, v, d, n, log, epsi)
            
def stock_embed_index(df, name=["embed_index"]):
    """
    """
    n_stocks = df["stock_id"].nunique()
    stock_map = dict(zip(df["stock_id"].unique(), list(range(n_stocks))))
    df.loc[:, name] = df["stock_id"].map(stock_map)
          
def final_feature_pipe(df, pipeline=[]):
    """
    """
    
    func_map= {
        "square_vars": square_vars,
        "interact_vars": interact_vars,
        "compute_ratio": compute_ratio,
        "stock_embed_index": stock_embed_index
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
        func(df, **args)                      

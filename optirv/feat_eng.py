
# ++++++++++++++++++++++++++ #
# Feature engineering module #
# ++++++++++++++++++++++++++ #

import numpy as np
import pandas as pd
from tqdm import tqdm
from optirv.data_loader import DataLoader
import optirv.pre_proc as pp

def add_real_vol_cols(base, df,
                      varnames=["WAP_lnret"], 
                      group_cols=["stock_id", "time_id"],
                      intervals = None):
    """
    """
    
    if intervals is None:
        intervals = [(0, 600)]

    for v in varnames:
        for i in intervals:
            rvol = df.query("@i[0] <= sec < @i[1]").groupby(
                group_cols, observed=True)[v].apply(pp.realized_vol)
            if intervals is None:
                new_name = v + "_vol"
            else:
                new_name = v + "_vol_%d_%d"%i        
            base = base.join(rvol.rename(new_name), on=group_cols)
        
    return base

def feat_eng_pipeline(data_mode="train", data_dir=None, 
                      stock_list=None, batch_size=3,
                      pp_merge_book_trade={"full_frame": True, "impute": True},
                      pp_compute_WAP=True,
                      pp_compute_lnret=(True, {"varnames": ["WAP"]}), 
                      fe_add_real_vol_cols=(True, {"intervals": None})):
    """
    """
    
    dl = DataLoader(data_mode=data_mode, data_dir=data_dir)
    if stock_list is None:
        stock_list = dl.target_df["stock_id"].unique().tolist()
    df_list = []
    
    for b in tqdm(range(int(np.ceil(len(stock_list)/batch_size)))):
        
        # load data for current batch of stocks
        stocks = stock_list[(b*batch_size):((b+1)*batch_size)]
        dl.pick_stocks(mode="specific", stocks=stocks)
        base = dl.target_df.query("stock_id in @dl.sample_stocks").copy()
        book_df, trade_df = dl.load_parquet()
        
        # merge book and trade data
        df = pp.merge_book_trade(book_df, trade_df, **pp_merge_book_trade)
        
        # Compute WAP and log returns for the order book
        if pp_compute_WAP:
            pp.compute_WAP(df)
        if pp_compute_lnret[0]:
            pp.compute_lnret(df, **pp_compute_lnret[1])
        
        # Start engineering features
        if fe_add_real_vol_cols[0]:
            base = add_real_vol_cols(base, df, **fe_add_real_vol_cols[1])
            
        # Append finalized batch data
        df_list.append(base)
        
    return pd.concat(df_list, ignore_index=True)
        
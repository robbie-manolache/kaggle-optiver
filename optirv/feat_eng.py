
# ++++++++++++++++++++++++++ #
# Feature engineering module #
# ++++++++++++++++++++++++++ #

import numpy as np
import pandas as pd
from tqdm import tqdm
from optirv.data_loader import DataLoader
import optirv.pre_proc as pp

def add_real_vol_cols(tgt_df, book_df,
                      varname="WAP_lnret", 
                      group_cols=["stock_id", "time_id"],
                      intervals = None):
    """
    """
    
    if intervals is None:
        intervals = [(0, 600)]

    for i in intervals:
        rvol = book_df.query("@i[0] <= sec < @i[1]").groupby(
            group_cols, observed=True)[varname].apply(pp.realized_vol)        
        tgt_df = tgt_df.join(rvol.rename("rvol_%d_%d"%i), on=group_cols)
        
    return tgt_df

def feat_eng_pipeline(data_mode="train", data_dir=None, 
                      stock_list=None, batch_size=3,
                      pp_merge_book_trade={"full_frame": True, "impute": True},
                      pp_compute_WAP=True,
                      pp_compute_lnret=(True, {"varname": "WAP"}), 
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
        tgt_df = dl.target_df.query("stock_id in @dl.sample_stocks").copy()
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
            tgt_df = add_real_vol_cols(tgt_df, df, 
                                       **fe_add_real_vol_cols[1])
        df_list.append(tgt_df)
        
    return pd.concat(df_list, ignore_index=True)
        
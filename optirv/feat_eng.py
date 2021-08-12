
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
        rvol = book_df.query("@i[0] <= seconds_in_bucket < @i[1]").groupby(
            group_cols, observed=True)[varname].apply(pp.realized_vol)        
        tgt_df = tgt_df.join(rvol.rename("rvol_%d_%d"%i), on=group_cols)
        
    return tgt_df

def feat_eng_pipeline(stock_list, batch_size=3, intervals=None):
    """
    """
    
    dl = DataLoader()
    df_list = []
    
    for b in tqdm(range(int(np.ceil(len(stock_list)/batch_size)))):
        
        # load data for current batch of stocks
        stocks = stock_list[(b*batch_size):((b+1)*batch_size)]
        dl.pick_stocks(mode="specific", stocks=stocks)
        batch_df = dl.target_df.query("stock_id in @dl.sample_stocks").copy()
        book_df, trade_df = dl.load_parquet()
        
        # Compute WAP and log returns for the order book
        pp.compute_WAP(book_df)
        pp.compute_lnret(book_df)
        
        # Start engineering features
        batch_df = add_real_vol_cols(batch_df, book_df, intervals=intervals)
        df_list.append(batch_df)
        
    return pd.concat(df_list, ignore_index=True)
        
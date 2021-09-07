
# +++++++++++++++++++++++ #
# Module for loading data #
# +++++++++++++++++++++++ #

import os 
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import optirv.pre_proc as pp

def __read_pq_data__(path, stocks):
    """
    """
    pq_con = pq.ParquetDataset(path, filters=[("stock_id", "in", stocks)])
    return pq_con.read().to_pandas().rename(columns={"seconds_in_bucket": "sec"}
                                            ).astype({'stock_id': 'int64'})

class DataLoader:
    
    def __init__(self, data_mode="train", data_dir=None):
        """
        data_mode:  "train" or "test"
        data_dir:   root data directory; if None, will look for DATA_DIR in 
                    enviroment variables
        """
        if data_dir is None:
            data_dir = os.environ.get("DATA_DIR")
        self.data_dir = data_dir
        self.target_df = pd.read_csv(os.path.join(data_dir, "%s.csv"%data_mode))
        self.book_dir = os.path.join(data_dir, "book_%s.parquet"%data_mode)
        self.trade_dir = os.path.join(data_dir, "trade_%s.parquet"%data_mode)
        self.sample_stocks = []
        
    def pick_stocks(self, mode="random", n=2, stocks=None, refresh=True):
        """
        Pick a subset of stocks to analyse
        mode:       "random" - randomly select n stocks
                    "specific" - specified by user using stocks arg
                    "all" - all stocks
        n:          number of stocks to randomly sample if mode is "random"
        stocks:     list of stock id integers that user wishes to pick
        refresh:    if True, current list of sample_stocks will be reset
        """
        
        if refresh:
            self.sample_stocks = []
            
        if mode == "random":
            sub_df = self.target_df[["stock_id"]].drop_duplicates()
            if len(self.sample_stocks) > 0:
                sub_df = sub_df.query("stock_id not in @self.sample_stocks")
            if n > sub_df.shape[0]:
                n = sub_df.shape[0]
            self.sample_stocks += sub_df.sample(n)["stock_id"].to_list()
            
        elif mode == "specific":
            self.sample_stocks += stocks
        
        elif mode == "all":
            self.sample_stocks = list(self.target_df["stock_id"].unique())
        
        else:
            print("Input to mode argument not recognized!")
            
    def load_parquet(self):
        """
        Loads parquet data for selected stocks
        """
        
        if len(self.sample_stocks) > 0:
            
            stocks = set(self.sample_stocks)
            book_df, trade_df = [__read_pq_data__(path, stocks) for path in
                                 (self.book_dir, self.trade_dir)]
            return book_df, trade_df
        
        else:
            
            print("Select stocks first using pick_stocks method!")
            return      
    
    def batcher(self, stock_list=None, batch_size=3):
        """
        Yields the base_df, book_df, trade_df for a set of stocks in batches.
        
        ARGS
          stock_list: list of stocks to consider. If None, all stocks selected.
          batch_size: number of stocks to process in each batch.
        """        
        
        if stock_list is None:
            stock_list = self.target_df["stock_id"].unique().tolist()
            
        for b in range(int(np.ceil(len(stock_list)/batch_size))):
            
            # load data for current batch of stocks
            stocks = stock_list[(b*batch_size):((b+1)*batch_size)]
            self.pick_stocks(mode="specific", stocks=stocks)
            base_df = self.target_df.query("stock_id in @self.sample_stocks").copy()
            book_df, trade_df = self.load_parquet()
            
            yield base_df, book_df, trade_df
                           
    def load_and_preproc(self, add_target=True,
                         pp_merge_book_trade={"full_frame": True, "impute": True},
                         pp_compute_WAP=True,
                         pp_compute_lnret=(True, {"varnames": ["WAP"]})):
        """
        """
        
        # load then combine book and trade data
        book_df, trade_df = self.load_parquet()
        df = pp.merge_book_trade(book_df, trade_df, **pp_merge_book_trade)
        
        # add target if required
        if add_target:
            tgt_df = self.target_df.query("stock_id in @self.sample_stocks")
            df = df.merge(tgt_df, on=["stock_id", "time_id"])
            
        # Compute WAP and log returns for the order book
        if pp_compute_WAP:
            pp.compute_WAP(df)
        if pp_compute_lnret[0]:
            pp.compute_lnret(df, **pp_compute_lnret[1])
        
        return df               
        
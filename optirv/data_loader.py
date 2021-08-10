
# +++++++++++++++++++++++ #
# Module for loading data #
# +++++++++++++++++++++++ #

import os 
import pandas as pd
import pyarrow.parquet as pq

def __read_pq_data__(path, stocks):
    """
    """
    pq_con = pq.ParquetDataset(path, filters=[("stock_id", "in", stocks)])
    return pq_con.read().to_pandas()

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
                           
            
        
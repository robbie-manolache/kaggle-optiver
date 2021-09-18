
# =-=-=-=-=-=-=-=-=-= # =-=-=-=-=-=-=-=-=-= #
# Feature aggregation # =-=-=-=-=-=-=-=-=-= #
# =-=-=-=-=-=-=-=-=-= # =-=-=-=-=-=-=-=-=-= #

import numpy as np

def agg_by_time_id(df, agg_vars, agg_func="mean", suffix="_agg"):
    """
    """
    for av in agg_vars:
        col = av + suffix 
        df.loc[:, col] = df.groupby("time_id")[av].transform(agg_func)    

def gen_distribution_stats(base,
                        dist_unit = ["stock_id"],
                        var_names=["no_ob_changes", "ob_time_length_med",
                                    "q_spread1_tw", "q_spread2_tw",
                                    "total_trades", "trade_size_med"],
                        percentile_spec=[50]):
    """
    base is at stock-id/time-id
    
    """
    stock_cs = base[["stock_id"]].drop_duplicates()
    
    for v in var_names:
        mean_name = v + "_mean"
        std_dev_name = v + "_std"
        
        mean = base.groupby(dist_unit, observed=True)[v].mean().rename(mean_name)
        std_dev = base.groupby(dist_unit, observed=True)[v].std().rename(std_dev_name)

        stock_cs = stock_cs.join(mean, on=dist_unit)
        stock_cs = stock_cs.join(std_dev, on=dist_unit)

        for i in percentile_spec:
            pct_temp = base.groupby(dist_unit, observed=True)[v].apply(
                lambda x: np.percentile(x.dropna(),i)).rename(v + "_pct_%d"%i)
            stock_cs = stock_cs.join(pct_temp, on=dist_unit)

    return stock_cs

def gen_outliers_threshold(trade_df,
                     dist_unit = ["stock_id"],
                     var_names=["size"],
                     percentile_spec=[98, 99]):
    """
    base is at trade_df
    
    """
    df = trade_df[["stock_id"]].drop_duplicates()
    
    for v in var_names:
        for i in percentile_spec:
            pct_temp = trade_df.groupby(dist_unit, observed=True)[v].apply(
                       lambda x: np.percentile(x.dropna(),i)).rename(v + "_pct_%d"%i)
            df = df.join(pct_temp, on=dist_unit)

    return df
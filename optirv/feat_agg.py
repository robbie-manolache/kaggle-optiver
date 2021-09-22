
# =-=-=-=-=-=-=-=-=-= # =-=-=-=-=-=-=-=-=-= #
# Feature aggregation # =-=-=-=-=-=-=-=-=-= #
# =-=-=-=-=-=-=-=-=-= # =-=-=-=-=-=-=-=-=-= #

import numpy as np
from sklearn.cluster import KMeans

def cluster_stocks(df, n_K=5, seed=31,
                   features=["nsec_w_trades", 
                             "WAP1_lnret_vol_all", "WAP1_lnret_BPV",
                             "q_spread1_tw", "q_spread2_tw", 
                             "q_spread1_std"],
                   stock_31=True):
    """
    """
    if stock_31:
        n_K += -1
    k_df = df.groupby("stock_id")[features].mean().reset_index()
    kmod = KMeans(n_clusters=n_K, random_state=31).fit(k_df[features])
    k_df.loc[:, "K"] = kmod.labels_
    if stock_31:
        k_df.loc[k_df["stock_id"]==31, "K"] = n_K
        n_K += 1    
    k_dict = {K: k_df.query("K == @K")["stock_id"].tolist() 
              for K in range(n_K)}
    
    return k_dict, k_df
    
def agg_by_time_id(df, agg_vars, agg_func="mean", suffix="_agg"):
    """
    """
    for av in agg_vars:
        col = av + suffix 
        df.loc[:, col] = df.groupby("time_id")[av].transform(agg_func)    

def agg_by_stock_id(df, agg_vars, agg_func="mean", suffix="_stock"):
    """
    """
    for av in agg_vars:
        col = av + suffix
        df.loc[:, col] = df.groupby("stock_id")[av].transform(agg_func)

def gen_distribution_stats(base, dist_unit=["stock_id"],
                           var_names=["no_ob_changes", "ob_time_length_med",
                                      "q_spread1_tw", "q_spread2_tw",
                                      "total_trades", "trade_size_med"],
                           excl_vars=["stock_id", "time_id", "target", "target_std",
                                      "target_chg", "weight", "segment"],
                           percentile_spec=[50]):
    """
    base is at stock-id/time-id
    
    """
    stock_cs = base[["stock_id"]].drop_duplicates()
    
    if var_names is None:
        var_names = [c for c in base.columns if c not in excl_vars]
    
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

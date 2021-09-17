
# =-=-=-=-=-=-=-=-=-= # =-=-=-=-=-=-=-=-=-= #
# Feature aggregation # =-=-=-=-=-=-=-=-=-= #
# =-=-=-=-=-=-=-=-=-= # =-=-=-=-=-=-=-=-=-= #



def agg_by_time_id(df, agg_vars, agg_func="mean", suffix="_agg"):
    """
    """
    for av in agg_vars:
        col = av + suffix 
        df.loc[:, col] = df.groupby("time_id")[av].transform(agg_func)    

def gen_distribution_stats(dist_base, df, 
                            var_names=["no_ob_changes", "ob_time_length_med",
                                        "q_spread1_tw", "q_spread2_tw",
                                        "total_trades", "trade_size_med"],
                            percentile_spec=[50]):
    """
    df = aggregate at stock_id-time_id
    returning distribution characteristics of a variable, default to by stock

    """
    
    for v in var_names:
        mean_name = v + "_mean"
        std_dev_name = v + "_std"
        
        mean = df.groupby(dist_unit, observed=True)[v].apply(lambda x: x.mean()).rename(mean_name)
        std_dev = df.groupby(dist_unit, observed=True)[v].apply(lambda x: x.std()).rename(std_dev_name)

        dist_base = dist_base.join(mean, on=dist_unit)
        dist_base = dist_base.join(std_dev, on=dist_unit)

        for i in percentile_spec:
            pct_temp = df.groupby(dist_unit, observed=True)[v].apply(lambda x: np.percentile(x.dropna(),i)).rename(v + "pct_%d"%i)
            dist_base = dist_base.join(pct_temp, on=dist_unit)

    return dist_base
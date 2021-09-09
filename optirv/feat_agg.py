
# =-=-=-=-=-=-=-=-=-= # =-=-=-=-=-=-=-=-=-= #
# Feature aggregation # =-=-=-=-=-=-=-=-=-= #
# =-=-=-=-=-=-=-=-=-= # =-=-=-=-=-=-=-=-=-= #



def agg_by_time_id(df, agg_vars, agg_func="mean", suffix="_agg"):
    """
    """
    for av in agg_vars:
        col = av + suffix 
        df.loc[:, col] = df.groupby("time_id")[av].transform(agg_func)    
    
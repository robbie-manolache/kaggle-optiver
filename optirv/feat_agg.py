
# =-=-=-=-=-=-=-=-=-= # =-=-=-=-=-=-=-=-=-= #
# Feature aggregation # =-=-=-=-=-=-=-=-=-= #
# =-=-=-=-=-=-=-=-=-= # =-=-=-=-=-=-=-=-=-= #



def agg_by_time_id(df, agg_vars, new_names,
                   agg_func="mean"):
    """
    """
    for a, n in zip(agg_vars, new_names):
        df.loc[:, n] = df.groupby("time_id")[a].transform(agg_func)
    
    
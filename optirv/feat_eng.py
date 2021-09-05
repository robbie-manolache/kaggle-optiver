
# ++++++++++++++++++++++++++ #
# Feature engineering module #
# ++++++++++++++++++++++++++ #

import numpy as np
import pandas as pd

def realized_vol(ln_ret_series, subset="all"):
    """
    subset:     Must be "all", "pos" or "neg"
    """
    if subset == "pos":
        ln_ret_series = ln_ret_series * (ln_ret_series > 0)
    elif subset == "neg":
        ln_ret_series = ln_ret_series * (ln_ret_series < 0)
    
    return np.sum((ln_ret_series ** 2))

def pre_retquad(ln_ret_series):
    """
    calculate returns to the power of 4
    """
    return np.sqrt(np.sum((ln_ret_series ** 4)) * len(ln_ret_series)/3)

def pre_compute_BPV(ln_ret_series):
    """
    """
    return np.sum(ln_ret_series * ln_ret_series.shift(-1))

def add_real_vol_cols(base, df, weights=None,
                      varnames=["WAP1_lnret", "WAP2_lnret"], 
                      group_cols=["stock_id", "time_id"],
                      subset="all",
                      interval_col="segment",
                      intervals=None):
    """
    df = book_df
    weights:    pandas.DataFrame containing a "weight" column and the same
                group_cols and interval_col as df. For a given interval, the 
                realized volatility will be divided by the sum of weights to 
                normalize it to the length of the entire time_id so that it 
                can be comparable no matter the length of the sub-period 
                captured by the interval.
                
    NOTE:   By default, the function uses "segment" as the interval_col, which
            is generated by the gen_segments() function. When using this function, 
            set return_segment_weights=True to generate a pandas.DataFrame that 
            can be passed to the weights argument.
    """
    
    if intervals is None:
        intervals = [(df[interval_col].min(), df[interval_col].max())]

    for v in varnames:
        for i in intervals:
            
            # derive volatility var name
            if intervals is None:
                rvol_name = "%s_vol_%s"%(v, subset)
                            
            else:
                rvol_name = "%s_vol_%s_%d_%d"%(v, subset, i[0], i[1])
                
            # compute realized volatility for each sub-segment of a time_id
            rvol = df.query("@i[0] <= %s <= @i[1]"%interval_col).groupby(
                group_cols, observed=True)[v].apply(realized_vol, 
                subset=subset).rename(rvol_name)

            # if weights provided, make sure volatility is on the same time scale
            if weights is not None:
                wgt = weights.query("@i[0] <= %s <= @i[1]"%interval_col).groupby(
                    group_cols, observed=True)[["weight"]].sum()
                wgt = wgt.join(rvol, on=group_cols)
                rvol = wgt[rvol_name] / wgt["weight"]
                rvol = rvol.rename(rvol_name)
                  
            base = base.join(rvol, on=group_cols)
        
    return base


def compute_BPV_retquad(base, df, weights=None,
                        varnames=["WAP1_lnret"], 
                        group_cols=["stock_id", "time_id"],
                        interval_col="segment",
                        intervals=None):
    """
    df = book_df
    weights:    pandas.DataFrame containing a "weight" column and the same
                group_cols and interval_col as df. For a given interval, the 
                realized volatility will be divided by the sum of weights to 
                normalize it to the length of the entire time_id so that it 
                can be comparable no matter the length of the sub-period 
                captured by the interval.
                
    NOTE:   By default, the function uses "segment" as the interval_col, which
            is generated by the gen_segments() function. When using this function, 
            set return_segment_weights=True to generate a pandas.DataFrame that 
            can be passed to the weights argument.
    """
    
    if intervals is None:
        intervals = [(df[interval_col].min(), df[interval_col].max())]

    for v in varnames:

        abs_var = v + "_abs"

        for i in intervals:
            
            # derive BPV var name
            if intervals is None:
                BPV_name = v + "_BPV"
                BPV_jump = v + "_BPV" + "_jump"
                RQ_name = v + "_RQ"
                rvol_name = v + "_vol_all"
            else:
                BPV_name = v + "_BPV_%d_%d"%(i[0], i[1])
                BPV_jump = v + "_BPV_%d_%d"%(i[0], i[1]) + "_jump"
                RQ_name = v + "_RQ_%d_%d"%(i[0], i[1])
                rvol_name = v + "_vol_all_%d_%d"%(i[0], i[1]) 

            # check rvol in base
            if rvol_name not in base.columns:
                print("%d needs to be computed"%rvol_name)
                return

            # compute BPV and ret_quad for each sub-segment of a time_id
            BPV = df.query("@i[0] <= %s <= @i[1]"%interval_col).groupby(
                group_cols, observed=True)[abs_var].apply(pre_compute_BPV).rename(BPV_name)

            RQ = df.query("@i[0] <= %s <= @i[1]"%interval_col).groupby(
                group_cols, observed=True)[v].apply(pre_retquad).rename(RQ_name)

            # if weights provided, make sure volatility is on the same time scale
            if weights is not None:
                wgt = weights.query("@i[0] <= %s <= @i[1]"%interval_col).groupby(
                    group_cols, observed=True)[["weight"]].sum()
                wgt = wgt.join(BPV, on=group_cols)
                wgt = wgt.join(RQ, on=group_cols)

                BPV = wgt[BPV_name] * (1 / wgt["weight"]) * (np.pi/2)
                BPV = BPV.rename(BPV_name)

                RQ = wgt[RQ_name] * (1 / wgt["weight"])
                RQ = RQ.rename(RQ_name)
                  
            base = base.join(BPV, on=group_cols)
            base = base.join(RQ, on=group_cols)
            base.loc[:, BPV_jump] = np.where((base[rvol_name] - base[BPV_name]) < 0, 
                                             0, (base[rvol_name] - base[BPV_name]))
        
    return base

def gen_weighted_var(base, df, equal_weight = False, 
                     var_names = ["slope_ask", "slope_bid", "q_spread1",
                                "q_spread2", "ratio_askP", "ratio_bidP"], 
                     group_cols = ["stock_id", "time_id"], 
                     weight_var = "time_length"):
    """
    df = book_df
    generating aggregated variable weighted by time_length
    also run this for base, df = merged book and trade, equal_weight = True,
                      var_names = ["ratio_size_depth1", "ratio_size_depth2"]
                   
    """

    for v in var_names:
        if equal_weight == False:
            weighted_var_name = v + "_tw"
            weighted_var = df.groupby(group_cols, observed = True)[[v, weight_var]].\
                apply(lambda x: np.sum(x[v] * x[weight_var])/np.sum(x[weight_var])).rename(weighted_var_name)
        
        else:
            weighted_var_name = v + "_ew"
            weighted_var = df.groupby(group_cols, observed = True)[v].\
                apply(lambda x: x.mean()).rename(weighted_var_name)
        
        base = base.join(weighted_var, on = group_cols)

    return base

def gen_last_obs(base, df, n_rows=1,
                var_names = ["ln_depth_total", "ratio_depth1_2",
                            "ratio_a_bdepth1","ratio_a_bdepth2",
                            "q_spread1", "q_spread2",
                            "ratio_askP", "ratio_bidP"],
                group_cols = ["stock_id", "time_id"]):
    """
    df = book_df
    generating the last observations for each stock_id-time_id
    """
    for v in var_names:
        
        last_var_name = v + "_last"
        
        if n_rows > 1:
            last_var_name += (str(n_rows) + "_avg")
        
        last_var = df.groupby(group_cols, observed = True)[group_cols + [v]]\
            .tail(n_rows).rename(columns={v: last_var_name})
        
        if n_rows > 1:
            last_var = last_var.groupby(group_cols, observed=True)[last_var_name]\
                .mean().reset_index()
        
        base = base.merge(last_var, on = group_cols)
    
    return base

def gen_trade_stats(base, df,
                    var_names = ["trade_size", "time_length"],
                    group_cols = ["stock_id", "time_id"]):
    """
    df = trade_df
    """
    sum_order_count = df.groupby(group_cols, observed = True)["order_count"].\
                    apply(lambda x: x.sum()).rename("total_trades")
    base = base.join(sum_order_count, on = group_cols)

    for v in var_names:
        median_var_name = v + "_med"
        max_var_name = v + "_max"
        median_var = df.groupby(group_cols, observed = True)[v].\
                    apply(lambda x: x.median()).rename(median_var_name)
        max_var = df.groupby(group_cols, observed = True)[v].\
                    apply(lambda x: x.max()).rename(max_var_name)
        
        base = base.join(median_var, on = group_cols)
        base = base.join(max_var, on = group_cols)
    
    return base

def gen_var_relto_dist(base, dist_df,
                       dist_unit = ["stock_id"],
                       dist_percentile = [90],
                       var_names = ["ln_depth_total_last", "ratio_depth1_2_last",
                                    "ratio_a_bdepth1_last", "ratio_a_bdepth2_last",
                                    "q_spread1_last", "q_spread2_last",
                                    "ratio_askP_last", "ratio_bidP_last",
                                    "trade_size_med", "time_length_med",
                                    "ratio_size_depth1_ew", "ratio_size_depth2_ew"]):
    """
    THIS IS NOT NEEDED
    dist_percentile has to be created first from pp.gen_distribution_stats
    """
    for v in var_names:
        for i in dist_percentile:
            dist_var_name = v + "pct_%d"%i
            new_var_name = v + "_relt" + "pct_%d"%i

            dist_var = dist_df[dist_unit + [dist_var_name]]
            base = base.merge(dist_var, on = dist_unit)
            base.loc[:, new_var_name] = base[v]/base[dist_var_name]
            base = base.drop(dist_var_name, axis = 1)

    return base    
        
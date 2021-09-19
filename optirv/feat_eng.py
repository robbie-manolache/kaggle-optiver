
# ++++++++++++++++++++++++++ #
# Feature engineering module #
# ++++++++++++++++++++++++++ #

import numpy as np
import pandas as pd

def add_real_vol_cols(base, df, weights=None,
                      varnames=["WAP1_lnret", "WAP2_lnret"], 
                      group_cols=["stock_id", "time_id"],
                      subsets=["all", "pos", "neg"]):
    """
    df = book_df
    weights:    pandas.DataFrame containing a "weight" column and the same
                group_cols and interval_col as df. For a given interval, the 
                realized volatility will be divided by the sum of weights to 
                normalize it to the length of the entire time_id so that it 
                can be comparable no matter the length of the sub-period 
                captured by the interval.

    ALSO RUN THIS FOR
        df = merged_book_trade
        varnames = ["WAP1_lnret_adj_99", "WAP1_lnret_adj_95"]

    """

    for v in varnames:
        for sub in subsets:
            
            # derive volatility var name
            q_df = df.copy() 
            rvol_name = "%s_vol_%s"%(v, sub)                          

            # multiplying filter for pos/neg returns
            if sub == "pos":
                mult = (q_df[v] > 0)
            elif sub == "neg":
                mult = (q_df[v] < 0)
            else:
                mult = 1
                
            # compute square returns
            q_df.loc[:, v+"_sq"] = (q_df[v] * mult) ** 2
                
            # compute realized volatility for each sub-segment of a time_id
            rvol = q_df.groupby(group_cols, observed=True)[v+"_sq"].sum(
                ).rename(rvol_name)

            # if weights provided, make sure volatility is on the same time scale
            if weights is not None:
                
                wgt = weights.groupby(group_cols, observed=True)[["weight"]].sum()
                wgt = wgt.join(rvol, on=group_cols)
                rvol = wgt[rvol_name] / wgt["weight"]
                rvol = rvol.rename(rvol_name)
                
            base = base.join(rvol, on=group_cols)
    
    return base


def compute_BPV_retquad(base, df, weights=None,
                        varnames=["WAP1_lnret"], 
                        group_cols=["stock_id", "time_id"]):
    """
    df = book_df
    weights:    pandas.DataFrame containing a "weight" column and the same
                group_cols and interval_col as df. For a given interval, the 
                realized volatility will be divided by the sum of weights to 
                normalize it to the length of the entire time_id so that it 
                can be comparable no matter the length of the sub-period 
                captured by the interval.
                
    ALSO run this for df =  merged book trade
                      varnames=["WAP1_lnret_adj_99", "WAP1_lnret_adj_95"]
    
    """
        
    for v in varnames:

        abs_var = v + "_abs"
        df.loc[:, abs_var] = np.abs(df[v])
        keep_cols = group_cols + ["sec", v, abs_var]

        # derive BPV var name
        BPV_name = v + "_BPV"
        BPV_jump = v + "_BPV" + "_jump"
        RQ_name = v + "_RQ"
        rvol_name = v + "_vol_all"
        q_df = df[keep_cols].copy()

            
        # check rvol in base
        if rvol_name not in base.columns:
            print("%d needs to be computed"%rvol_name)
            return
        
        # get the last second by stock and time id
        q_df.loc[:, "max_sec"] = q_df.groupby(["stock_id", "time_id"], 
                                                observed=True)["sec"].transform("max")
        # shift the absolute return var f1
        q_df.loc[:, abs_var+"_f1"] = q_df[abs_var].shift(-1)
        
        # NA any last obs in a stock-time segment as it came from another segment
        q_df.loc[q_df["sec"]==q_df["max_sec"], abs_var+"_f1"] = np.nan
        
        # Finally, compute the product between shifted and absolute returns
        q_df.loc[:, abs_var+"_cov"] = q_df[abs_var] * q_df[abs_var+"_f1"]
        
        # Compute quad returns
        q_df.loc[:, v+"_quad"] = (q_df[v] ** 4)
        
        # compute BPV and ret_quad for each sub-segment of a time_id
        BPV = q_df.groupby(group_cols, observed=True)[abs_var+"_cov"].sum().\
              rename(BPV_name) * (np.pi/2)
        RQ = q_df.groupby(group_cols, observed=True)[v+"_quad"].sum()
        N_grp = q_df.groupby(group_cols, observed=True)[v+"_quad"].count()
        RQ = np.sqrt(RQ * N_grp / 3).rename(RQ_name)

        # if weights provided, make sure volatility is on the same time scale
        if weights is not None:
            
            wgt = weights.groupby(group_cols, observed=True)[["weight"]].sum()
            wgt = wgt.join(BPV, on=group_cols)
            wgt = wgt.join(RQ, on=group_cols)

            BPV = wgt[BPV_name] * (1 / wgt["weight"])
            BPV = BPV.rename(BPV_name)

            RQ = wgt[RQ_name] * (1 / wgt["weight"])
            RQ = RQ.rename(RQ_name)
                
        base = base.join(BPV, on=group_cols)
        base = base.join(RQ, on=group_cols)
        base.loc[:, BPV_jump] = np.where((base[rvol_name] - base[BPV_name]) < 0, 
                                            0, (base[rvol_name] - base[BPV_name]))
        
    return base

def gen_weighted_var(base, df, equal_weight = False,
                    first_time_book = True,
                    var_names = ["slope_ask", "slope_bid", 
                                  "q_spread1", "q_spread2", 
                                  "ask_spread", "bid_spread",
                                  "bid_size1", "ask_size1",
                                  "bid_size2", "ask_size2",
                                  "ratio_depth1_2", "depth_imb1", "depth_imb_total"], 
                    group_cols = ["stock_id", "time_id"], 
                    weight_var = "time_length"):
    """
    df = book_df
    generating aggregated variable weighted by time_length
    also run this for base, df = merged book and trade, 
                            equal_weight = True,
                            first_time_book = False,
                            var_names = ["ratio_size_depth1", "ratio_size_depth2"]              
    """
    if first_time_book:

        sum_ob_change = df.groupby(group_cols, observed = True)["sec"].\
                        count().rename("no_ob_changes")
        base = base.join(sum_ob_change, on = group_cols, how="left").fillna(0)

        sum_m1_change = df.groupby(group_cols, observed = True)["midquote1_diff"].\
                        sum().rename("no_m1_changes")
        base = base.join(sum_m1_change, on = group_cols, how="left").fillna(0)
    
    if equal_weight == False:
        df.loc[:, "weight_frac"] = df[weight_var] / df.groupby(
            group_cols, observed=True)[weight_var].transform("sum")
    
    for v in var_names:
        if equal_weight == False:
            weighted_var_name = v + "_tw"
            df.loc[:, "temp_var"] = df["weight_frac"] * df[v]
            weighted_var = df.groupby(group_cols, observed = True)["temp_var"].\
                sum().rename(weighted_var_name)
        
        else:
            weighted_var_name = v + "_ew"
            weighted_var = df.groupby(group_cols, observed = True)[v].\
                mean().rename(weighted_var_name).fillna(0)
        
        base = base.join(weighted_var, on = group_cols)

    if equal_weight == False:
        df.drop(columns=["weight_frac", "temp_var"], inplace=True)
    
    return base

def gen_st_dev(base, df,
               var_names = ["q_spread1", "q_spread2", 
                            "ask_spread", "bid_spread",
                            "ratio_depth1_2", "depth_imb1", "depth_imb_total"], 
               group_cols = ["stock_id", "time_id"]):
    """
    df = book_df
    """
    
    new_names = [v + "_std" for v in var_names]

    std_df = df.groupby(group_cols, observed=True)[var_names].std()
    std_df.columns = new_names

    base = base.join(std_df, on=group_cols)
    
    return base

def gen_last_obs(base, df, n_rows=1,
                var_names = ["q_spread1", "q_spread2", 
                            "ask_spread", "bid_spread",
                            "bid_size1", "ask_size1",
                            "bid_size2", "ask_size2",
                            "ratio_depth1_2", "depth_imb1", "depth_imb_total"],
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
                    fill_na = [0, 600],
                    group_cols = ["stock_id", "time_id"]):
    """
    df = trade_df
    NOTE: fill_na needs to be same length as var_names and each numeric entry will 
          be used to fill NA values for the corresponding variable
    """
    sum_order_count = df.groupby(group_cols, observed = True)["order_count"].\
                      sum().rename("no_trades")
    base = base.join(sum_order_count, on = group_cols, how="left").fillna(0)

    sum_quantity = df.groupby(group_cols, observed = True)["size"].\
                   sum().rename("total_trade_vol")
    base = base.join(sum_quantity, on = group_cols, how="left").fillna(0)

    sum_obs = df.groupby(group_cols, observed = True)["sec"].\
                   count().rename("nsec_w_trades")
    base = base.join(sum_obs, on = group_cols, how="left").fillna(0)

    for v, f in zip(var_names, fill_na):
        median_var_name = v + "_med"
        max_var_name = v + "_max"
        median_var = df.groupby(group_cols, observed = True)[v].\
            median().rename(median_var_name)
        max_var = df.groupby(group_cols, observed = True)[v].\
            max().rename(max_var_name)
        
        base = base.join(median_var, on = group_cols, how="left").fillna(f)
        base = base.join(max_var, on = group_cols, how="left").fillna(f)
    
    return base

def gen_adj_trade_stats(base, df,
                        var_names = ["order_count", "size"],
                        output_var = ["total_trades", "total_trade_vol"],
                        fill_na = 0,
                        group_cols = ["stock_id", "time_id"],
                        percentile_spec=[95, 99]):
    """
    df = merged book trade
    """

    for i in percentile_spec:
        for v, out_v in zip(var_names, output_var):

            sum = df.groupby(group_cols, observed = True)[v+"_adj_%d"%i].\
                        sum().rename(out_v+"_adj_%d"%i)
            base = base.join(sum, on = group_cols, how="left").fillna(fill_na)  
    
    return base

# ____ Depracated ____ #     

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
        

# ++++++++++++++++++++++++ #
# Module for basic visuals #
# ++++++++++++++++++++++++ #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from optirv.feat_eng import realized_vol

def plot_returns_by_time(base, book, stock_id, 
                         n_times=8, time_id=[],
                         return_cols=["WAP_lnret"],
                         include_rvol=True,
                         base_rvol_on=0):
    """
    """

    # time id selection
    if len(time_id) == 0:
        time_id = np.random.choice(base.query("stock_id == @stock_id")["time_id"], 
                                   size=n_times, replace=False).tolist()
    else:
        n_times = len(time_id)
        
    # plot data selection
    plot_df = book.query("stock_id == @stock_id & time_id in @time_id").copy()
    plot_df = plot_df[["time_id", "sec"] + return_cols]
    
    # add realized vol
    plot_df.loc[:, "rvol"] = plot_df.groupby(
        "time_id")[return_cols[base_rvol_on]].transform(
            lambda x: realized_vol(x, square_root=True))
    plot_df = plot_df.merge(base.query("stock_id == @stock_id"), on="time_id")
    if include_rvol:
        plot_cols = return_cols + ["rvol", "target"]    
    
    # create plot
    fig, axes = plt.subplots(int(np.ceil(n_times/2)), 2, 
                             figsize=(10, n_times*2))
    for (t, group), ax in zip(plot_df.groupby("time_id"), axes.flatten()):
        rvol, target = group[["rvol", "target"]].mean()
        title_txt = "Time: %d || Current: %.4f || Target: %.4f"%(t, rvol, target)
        group.plot(x='sec', y=plot_cols, kind='line', ax=ax, title=title_txt)
        ax.set_ylim(ymin=plot_df[plot_cols].min().min(), 
                    ymax=plot_df[plot_cols].max().max()*1.1)

    plt.tight_layout()    
    plt.show()
    
    

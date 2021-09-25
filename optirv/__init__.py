
# support functions
from optirv.helpers.config import env_config
from optirv.lazykaggler.competitions import competition_download, \
    competition_files, competition_list
from optirv.lazykaggler.kernels import kernel_output_download
from optirv.lazykaggler.datasets import gen_dataset_metafile, upload_dataset

# main functions
from optirv.data_loader import DataLoader
from optirv.pre_proc import merge_book_trade, compute_WAP, compute_lnret, \
    gen_ob_slope, gen_ob_var, gen_merged_book_trade_var, gen_trade_var, \
    gen_segments_by_time, gen_segments_by_obs, gen_segment_weights, \
    gen_outliers_threshold, gen_outlier_flags, corr_multivar 
from optirv.feat_eng import add_real_vol_cols, compute_BPV_retquad, \
    gen_weighted_var, gen_last_obs, gen_trade_stats, gen_adj_trade_stats, \
    gen_st_dev, gen_depth_change, gen_var_relto_dist
from optirv.final_feats import adjust_vars, interact_vars, compute_ratio, \
    stock_embed_index, gen_target_class, gen_target_change, gen_weights, \
    add_stock_vars, standardize_by_stock, standardize_target, \
    reshape_segments, seg_based_change, seg_based_feats, seg_based_agg, \
    log_norm, cap_vars, drop_vars, gen_rv_outliers_flag, gen_rv_outliers 
from optirv.feat_agg import cluster_stocks, cluster_by_corr, agg_by_time_id, \
    agg_by_stock_id, gen_distribution_stats
from optirv.data_pipes import gen_seg_base, feat_eng_pipeline, final_feature_pipe
from optirv.data_viz import plot_returns_by_time, plot_fcst_vs_act, \
    confusion_matrix
from optirv.train_lgbm import train_lgbm_model, lgbm_CV
from optirv.train_regressor import build_NN_model, train_NN_model, regression_CV
from optirv.eval_tools import rmspe_calc, predict_target_class, multi_log_loss, \
    predict_target, adjust_preds, cv_reg_stats

"""
what to analyze
- performance curve
- performance area
- m-h, h-m and percentage
- pure accuracy
- robust accuracy (transformed accuracy)
- prediction similarity
"""

import shutil
import numpy as np
import pandas as pd
from typing import List, Dict, Union
from pathlib2 import Path
from tabulate import tabulate
from argparse import ArgumentParser
from scipy.interpolate import UnivariateSpline

from reliabilitycli.src.workspace import Workspace
from reliabilitycli.src.utils.metric import AnalysisResult, calculate_acc, calculate_accs, performance_area, compare_two_splines


def setup_analyze_parser(analyze_parser: ArgumentParser):
    analyze_parser.add_argument(
        '-s', '--sample_csv', type=str, default='sample_result.csv', help='filename of sample_csv (not path, must be in workspace)')
    analyze_parser.add_argument(
        '-r', '--eval_csv', type=str, default='eval_results.csv', help='filename of eval csv (not path, must be in workspace)')
    analyze_parser.add_argument(
        '--smoothing_factor', type=float, default=0.04, help='Smoothing Factor For Spline')
    analyze_parser.add_argument(
        '-g', '--graph', action='store_true', help='Generate Graphs')
    analyze_parser.add_argument(
        '-b', '--bins', type=int, default=20, help='Number of Bins')


def get_vd_bins_dfs(df: pd.DataFrame, n_bins: int):
    bins_dfs = []
    for i in range(n_bins):
        bins_dfs.append(df[(df['vd_score'] > i / n_bins) &
                        (df['vd_score'] < ((i + 1) / n_bins))])
    return bins_dfs


def analyze_both(w: Workspace, eval_csv_path: Path, human_eval_csv_path: Path):
    pass


def analyze_model_eval_df(model_eval_df: pd.DataFrame, n_bins: int, smoothing_factor: float):
    # first turn into vd_score bins
    bins_dfs = get_vd_bins_dfs(model_eval_df, n_bins)
    # compute raw accuracies
    raw_orig_accs = calculate_accs(bins_dfs, 'orig_pred', 'label')
    raw_transf_accs = calculate_accs(bins_dfs, 'transf_pred', 'label')
    raw_pred_sims = calculate_accs(bins_dfs, 'transf_pred', 'orig_pred')
    # compute splines
    xs = np.arange(n_bins) / n_bins
    orig_spl = UnivariateSpline(xs, raw_orig_accs, s=smoothing_factor)
    transf_spl = UnivariateSpline(xs, raw_transf_accs, s=smoothing_factor)
    pred_sim_spl = UnivariateSpline(xs, raw_pred_sims, s=smoothing_factor)
    # compute smoothed accuracies
    smoothed_orig_accs = orig_spl(xs)
    smoothed_transf_accs = transf_spl(xs)
    smoothed_pred_sims = pred_sim_spl(xs)
    # compute performance area under curve
    raw_orig_acc_area = performance_area(raw_orig_accs)
    raw_transf_acc_area = performance_area(raw_transf_accs)
    raw_pred_sim_area = performance_area(raw_pred_sims)
    smoothed_orig_acc_area = performance_area(smoothed_orig_accs)
    smoothed_transf_acc_area = performance_area(smoothed_transf_accs)
    smoothed_pred_sim_area = performance_area(smoothed_pred_sims)
    return AnalysisResult(n_bins, smoothing_factor, raw_orig_accs, raw_transf_accs, raw_pred_sims,
                          smoothed_orig_accs, smoothed_transf_accs, smoothed_pred_sims, raw_orig_acc_area,
                          raw_transf_acc_area, raw_pred_sim_area, smoothed_orig_acc_area, smoothed_transf_acc_area,
                          smoothed_pred_sim_area, orig_spl, transf_spl, pred_sim_spl)


def analyze_models(n_bins: int, smoothing_factor: float, eval_df: pd.DataFrame) -> Dict[str, AnalysisResult]:
    unique_models = list(eval_df['model'].unique())
    model_analysis_results_dict = {}
    for model_name in unique_models:
        model_df = eval_df[eval_df['model'] == model_name]
        model_analysis_results_dict[model_name] = analyze_model_eval_df(
            model_df, n_bins, smoothing_factor)
    return model_analysis_results_dict


def report_ai_vs_human(models_analysis_results_dict: Dict[str, AnalysisResult], human_analysis_result, mode: str):
    """
    mode should be either acc or pred_sim
    """
    data = []
    for model_name in models_analysis_results_dict.keys():
        model_analysis_result = models_analysis_results_dict[model_name]
        if mode == 'acc':
            a_ml, a_h, a_ml_h, a_h_ml, p_ml_h, p_h_ml = compare_two_splines(model_analysis_result.transf_spl, human_analysis_result.transf_spl)
        else:
            a_ml, a_h, a_ml_h, a_h_ml, p_ml_h, p_h_ml = compare_two_splines(model_analysis_result.pred_sim_spl, human_analysis_result.pred_sim_spl)
            
        data.append({
            "model": model_name,
            "A_ml": a_ml,
            "A_h": a_h,
            "A_ml-h": a_ml_h,
            "A_h-ml": a_h_ml,
            "P_ml-h": p_ml_h,
            "P_h-ml": p_h_ml
        })
    return pd.DataFrame(data)


def analyze(w: Workspace, eval_csv_path: Path, human_eval_csv_path: Path = None):
    if w.report_dir_path.exists():
        shutil.rmtree(w.report_dir_path)
    w.report_dir_path.mkdir(parents=True, exist_ok=True)
    ml_eval_df = pd.read_csv(eval_csv_path, index_col=0)
    models_analysis_results_dict = analyze_models(
        w.get_args().bins, w.get_args().smoothing_factor, ml_eval_df)
    ml_report_data = []
    ml_report_data = [models_analysis_results_dict[model_name].to_dict(
    ) for model_name in models_analysis_results_dict.keys()]
    for idx, model_name in enumerate(models_analysis_results_dict.keys()):
        ml_report_data[idx]['model'] = model_name
    ml_report_df = pd.DataFrame(ml_report_data)

    print(tabulate(ml_report_df, tablefmt='pretty', headers=ml_report_df.columns))
    ml_report_df.to_csv(w.get_workspace_path() / 'ml_report.csv')

    if human_eval_csv_path is not None and human_eval_csv_path.exists():
        human_eval_df = pd.read_csv(human_eval_csv_path)
        human_analysis_result = analyze_model_eval_df(
            human_eval_df, w.get_args().bins,  w.get_args().smoothing_factor)
        print("Compare Transformed Accuracy Spline with Human")
        df = report_ai_vs_human(models_analysis_results_dict, human_analysis_result, 'acc')
        df.to_csv(w.get_workspace_path() / 'ml_vs_human_acc_report.csv')
        print(tabulate(df, headers=df.columns, tablefmt='pretty'))
        print("Compare Prediction Similarity Spline with Human")
        df = report_ai_vs_human(models_analysis_results_dict, human_analysis_result, 'pred_sim')
        df.to_csv(w.get_workspace_path() / 'ml_vs_human_pred_sim_report.csv')
        print(tabulate(df, headers=df.columns, tablefmt='pretty'))


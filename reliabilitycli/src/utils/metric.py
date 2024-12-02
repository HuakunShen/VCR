import numpy as np
import pandas as pd
from typing import List, Dict
from scipy.interpolate import UnivariateSpline
from dataclasses import dataclass
from typing import Tuple


@dataclass
class AnalysisResult:
    n_bins: int
    smoothing_factor: float
    raw_orig_accs: List[float]
    raw_transf_accs: List[float]
    raw_pred_sims: List[float]
    smoothed_orig_accs: List[float]
    smoothed_transf_accs: List[float]
    smoothed_pred_sims: List[float]
    raw_orig_acc_area: float
    raw_transf_acc_area: float
    raw_pred_sim_area: float
    smoothed_orig_acc_area: float
    smoothed_transf_acc_area: float
    smoothed_pred_sim_area: float
    orig_spl: UnivariateSpline
    transf_spl: UnivariateSpline
    pred_sim_spl: UnivariateSpline

    def to_dict(self) -> Dict:
        return {
            "raw_orig_acc_area": round(self.raw_orig_acc_area, 5),
            "raw_transf_acc_area": round(self.raw_transf_acc_area, 5),
            "raw_pred_sim_area": round(self.raw_pred_sim_area, 5),
            "smoothed_orig_acc_area": round(self.smoothed_orig_acc_area, 5),
            "smoothed_transf_acc_area": round(self.smoothed_transf_acc_area, 5),
            "smoothed_pred_sim_area": round(self.smoothed_pred_sim_area, 5)
        }


def calculate_acc(df: pd.DataFrame, col1: str, col2: str) -> float:
    """Calculate accuracy based on 2 given columns"""
    return (df[col1] == df[col2]).sum() / len(df)


def calculate_accs(bins_dfs: List[pd.DataFrame], col1: str, col2: str):
    """Calculate accuracies based on 2 given columns"""
    prev_acc = 0
    accs = []
    for bin_df in bins_dfs:
        if len(bin_df) == 0:
            # if current vd bin has no data, reuse previous accuracy to avoid division by 0
            acc = prev_acc
        else:
            acc = calculate_acc(bin_df, col1, col2)
        accs.append(acc)
    return accs


def performance_area(accs: List[float]):
    x, y = [], []
    n_bins = len(accs)
    y_init = np.linspace(0, 1, n_bins)
    for i, acc in enumerate(accs):
        if not np.isnan(acc):
            x.append(y_init[i])
            y.append(acc)
    return np.trapz(y, x=x)


def spl_to_accs_monotonic(xs: np.array, spl: UnivariateSpline) -> np.array:
    ys = np.clip(spl(xs), 0, 1)
    for i in range(1, len(ys)):
        if ys[i] > ys[i - 1]:
            ys[i] = ys[i - 1]
    return ys


def compare_two_splines(spl1: UnivariateSpline, spl2: UnivariateSpline) -> Tuple[float, float, float, float, float, float]:
    """
    Given 2 spline models (e.g. ai and human), return the area difference
    if spl1 is ai spline and spl2 is human spl, then returned value is 
    - ai performance area
    - human performance area
    - ai minus human area
    - human minus ai human area
    - ai minus human percentage
    - human minus ai percentage
    """
    xs = np.arange(10000) / 10000
    spl1_ys = spl_to_accs_monotonic(xs, spl1)
    spl2_ys = spl_to_accs_monotonic(xs, spl2)
    spl_2_1_diff = np.where(spl2_ys > spl1_ys, spl2_ys - spl1_ys, 0)
    spl_1_2_diff = np.where(spl1_ys > spl2_ys, spl1_ys - spl2_ys, 0)
    spl_2_1_area = np.trapz(spl_2_1_diff, x=xs)
    spl_1_2_area = np.trapz(spl_1_2_diff, x=xs)

    area1 = np.trapz(spl1_ys, x=xs)
    area2 = np.trapz(spl2_ys, x=xs)

    p_1_2 = 1 - (spl_1_2_area / area1)
    p_2_1 = 1 - (spl_2_1_area / area2)
    return area1, area2, spl_1_2_area, spl_2_1_area, p_1_2, p_2_1

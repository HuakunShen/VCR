import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Iterable
from scipy.interpolate import UnivariateSpline


def spline(x: Iterable[float], y: Iterable[float], smoothing_factor: float = None):
    spl = UnivariateSpline(x, y)
    if smoothing_factor is not None:
        spl.set_smoothing_factor(smoothing_factor)
    return spl


def visualize_spline(data: pd.DataFrame, title: str):
    """
    Expect a data frame with columns:
    [type: (ml | human), acc: float, delta_v: float]
    """
    plt.ylim(0, 100)
    sns.lineplot(data, x="delta_v", y="acc", hue='type')
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    rng = np.random.default_rng()
    xs = np.linspace(0, 100, 100)
    sns.set_style("darkgrid")
    human_mean = 80
    human_data = human_mean + np.exp(-xs ** 2) + 10 * rng.standard_normal(100)
    ml_mean = 75
    ml_data = ml_mean + np.random.normal(0, 1, 100)
    data = {
        "type": ["ml"] * 100 + ["human"] * 100,
        "acc": list(ml_data) + list(human_data),
        "delta_v": list(xs) + list(xs)
    }
    df = pd.DataFrame(data)
    print(df)
    visualize_spline(df, 'vis1')

    human_spline = spline(xs, human_data, 0.5)
    ml_spline = spline(xs, ml_data, 0.5)
    data = {
        "type": ["ml"] * 100 + ["human"] * 100,
        "acc": list(ml_spline(xs)) + list(human_spline(xs)),
        "delta_v": list(xs) + list(xs)
    }
    visualize_spline(pd.DataFrame(data), 'vis2')


    # compute top and bottom diff
    xs = np.linspace(0, 100, 100000)
    human_ys = human_spline(xs)
    ml_ys = ml_spline(xs)
    human_top_ys = [max(human_ys[i], ml_ys[i]) for i in range(len(human_ys))]
    # ml_top_ys = [max(human_ys[i], ml_ys[i]) for i in range(len(human_ys))]
    # print(len(ml_ys))
    # print(len(human_top_ys))
    types = ["ml"] * 100000 + ["human"] * 100000
    acc = list(ml_ys) + list(human_top_ys)
    data = {
        "type": types,
        "acc": acc,
        "delta_v": xs.tolist() + xs.tolist()
    }
    df = pd.DataFrame(data)
    print(df)
    visualize_spline(df, 'vis3')




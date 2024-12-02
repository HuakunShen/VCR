import pandas as pd
from tabulate import tabulate


def table2str(df: pd.DataFrame, head: int = None):
    """
    Helper for transforming pandas dataframe to a nice-looking string format with tabulate
    :param df: pandas Data Frame
    :param head: number of rows to display, avoid displaying thousands of rows
    :return: table in str format generated with tabulate
    """
    if head is not None:
        df = df.head(head)
    return tabulate(df, headers=df.columns, tablefmt='pretty')

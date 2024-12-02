from __future__ import annotations

import logging
import pandas as pd
from pathlib2 import Path
from tabulate import tabulate
from abc import ABC, abstractmethod

from reliabilitycli.src.workspace import Workspace


class DatasetInfo(ABC):
    def __init__(self, data_root: Path):
        self.data_root = data_root.absolute()
        self._id = []
        self._width = []
        self._height = []
        self._filename = []
        self._path = []

    @property
    @abstractmethod
    def image_info_df(self) -> pd.DataFrame:
        """Should have columns
        - id
        - filename
        - path
        - width
        - height
        TODO: Consider which columns can be removed
        :raises NotImplementedError: [description]
        :return: pandas DataFrame with very basic image info data
        :rtype: pd.DataFrame
        """
        raise NotImplementedError

    @abstractmethod
    def parse(self) -> DatasetInfo:
        raise NotImplementedError

    def __str__(self, full: bool = False) -> str:
        """Turn Dataset Info into a table

        :param full: whether return full table, header might be invisible with full table, defaults to True
        :type full: bool, optional
        :return: table of datasets info in str
        :rtype: str
        """
        df = self.image_info_df.head() if not full else self.image_info_df
        return tabulate(df, tablefmt='pretty', headers=df.columns)

    @abstractmethod
    def verify_root_validity(self):
        """
        Verify that the data root is valid by checking whether the class directories all exist
        :return: valid or not
        """
        raise NotImplementedError

    # @property
    # def logger(self) -> logging.Logger:
    #     """
    #     an alias (shortcut) to get logger for current workspace (experiment)
    #     :return: a logger
    #     """
    #     return Workspace.instance().get_logger()

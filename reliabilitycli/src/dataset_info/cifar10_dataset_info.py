from __future__ import annotations

import torchvision
from typing import Union, List
from pathlib2 import Path

from reliabilitycli.src.constants.dataset import CIFAR10_CLASSES
from reliabilitycli.src.dataset_info import ClassificationDatasetInfo


class Cifar10DatasetInfo(ClassificationDatasetInfo):
    """
    Parse and store Cifar10 dataset information
    """    
    def __init__(self, data_root: Union[Path, str], train: bool = True, classes: List[str] = CIFAR10_CLASSES):
        self.data_root = data_root.absolute() / 'train' if train else data_root.absolute() / 'val'
        super(Cifar10DatasetInfo, self).__init__(self.data_root, classes)
        self.parse(ext='png')

from pathlib2 import Path

from reliabilitycli.src.constants import IMAGENET, CIFAR10
from reliabilitycli.src.constants.dataset import IMAGENET16CLASS
from reliabilitycli.src.dataset_info import Cifar10DatasetInfo, ImagenetDatasetInfo, ClassificationDatasetInfo
from reliabilitycli.src.dataset_info.imagenet_dataset_info import Imagenet16ClassDatasetInfo


def dataset_info_factory(dataset_name: str, dataset_dir: Path) -> ClassificationDatasetInfo:
    if dataset_name == IMAGENET:
        dataset_info = ImagenetDatasetInfo(data_root=dataset_dir / 'ImageNet' / 'val',
                                           original_data_root=dataset_dir / 'ImageNet',
                                           original_split='val')
    elif dataset_name == IMAGENET16CLASS:
        dataset_info = Imagenet16ClassDatasetInfo(data_root=dataset_dir / 'ImageNet' / 'val',
                                           original_data_root=dataset_dir / 'ImageNet',
                                           original_split='val')
    elif dataset_name == CIFAR10:
        dataset_info = Cifar10DatasetInfo(
            data_root=dataset_dir / 'cifar10' / 'cifar10_pytorch')
    else:
        raise ValueError(f"Invalid Dataset Name: {dataset_name}")
    return dataset_info

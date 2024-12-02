from reliabilitycli.src.dataset_info.dataset_info import DatasetInfo
from reliabilitycli.src.dataset_info.classification_dataset_info import ClassificationDatasetInfo
from reliabilitycli.src.dataset_info.imagenet_dataset_info import ImagenetDatasetInfo
from reliabilitycli.src.dataset_info.cifar10_dataset_info import Cifar10DatasetInfo
from reliabilitycli.src.dataset_info.coco_dataset_info import CocoDatasetInfo
from reliabilitycli.src.dataset_info.pascal_voc_dataset_info import PascalVOCDatasetInfo
from reliabilitycli.src.constants import IMAGENET, CIFAR10
from reliabilitycli.src.workspace import Workspace


def load_dataset_info():
    w = Workspace.instance()
    if w.config.config['dataset_name'] == IMAGENET:
        dataset_info = ImagenetDatasetInfo(data_root=w.config.dataset_dir / 'ImageNet' / 'val',
                                       original_data_root=w.config.dataset_dir / 'ImageNet',
                                       original_split='val')
    elif w.config.config['dataset_name'] == CIFAR10:
        dataset_info = Cifar10DatasetInfo(data_root=w.config.dataset_dir / 'cifar10' / 'cifar10_pytorch')
    else:
        # custom datasets, assume we get a image folder structure
        dataset_info = ClassificationDatasetInfo(data_root=w.config.dataset_dir / w.config.config['dataset_name'])
    return dataset_info

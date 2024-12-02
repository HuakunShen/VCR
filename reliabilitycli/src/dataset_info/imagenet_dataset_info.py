import torchvision.datasets as dst
from typing import Union, Dict
from pathlib2 import Path
import pandas as pd

from reliabilitycli.src.dataset_info import ClassificationDatasetInfo
from reliabilitycli.src.constants.imagenet_16_class_map import imagenet_16_class_wnid_to_category


class ImagenetDatasetInfo(ClassificationDatasetInfo):
    def __init__(self, data_root: Union[Path, str], original_data_root: Union[Path, str], original_split: str = 'val'):
        """
        2 data roots are required because I need to load ImageNet metadata such as classes and class-label mapping using
        the official PyTorch ImageNet class and original data root which contains the metadata. For each subset of
        ImageNet I created, there are only images. So data_root should be a directory of images, and original_data_root
        contains image folder such as "val" and other metadata files.
        :param data_root: data root to dataset where images will be loaded, a directory of images, can be a subset of the original dataset
        :param original_data_root: path to the original ImageNet dataset, for me this dataset contains the following files: ILSVRC2012_devkit_t12.tar.gz, ILSVRC2012_img_val.tar, meta.bin, val
        :param original_split: use val or train for original dataset
        """
        self.original_split = original_split
        self.full_data_root = Path(original_data_root).absolute()
        self.original_dataset = dst.ImageNet(original_data_root, split=self.original_split)
        self.wnids = self.original_dataset.wnids  # list of label id (e.g. 'n01440764'), len=1000
        self.wnid_to_idx = self.original_dataset.wnid_to_idx  # map wnids to idx (0-999)
        self.classes_in_english = self.original_dataset.classes  # List of tuple (len=1000), each tuple is labels in en
        self.classes = self.wnids
        super(ImagenetDatasetInfo, self).__init__(Path(data_root), self.classes)
        self.class_to_idx = self.original_dataset.class_to_idx  # map english label to idx (len==1842)
        self.parse(ext="JPEG")  # parse all files that ends with JPEG

    def class_to_id(self, _class: str) -> int:
        """The _class here should be wnid"""
        return self.wnid_to_idx[_class]

    @property
    def wnid2idx(self) -> Dict[str, int]:
        """
        Map ImageNet labels (e.g. n01440764) to index. The index we talk about here is not image id used in DatasetInfo.
        The id ranges betwen 0 and 999, represents the 1000 classes of ImageNet
        :return: a dict mapping ImageNet label to index as int
        """
        return self.wnid_to_idx


class Imagenet16ClassDatasetInfo(ImagenetDatasetInfo):
    """
    This is a subset of the original imagenet dataset
    """
    def __init__(self, data_root: Union[Path, str], original_data_root: Union[Path, str], original_split: str = 'val'):
        super().__init__(data_root, original_data_root, original_split)
        self.orig_class_to_idx = self.class_to_idx
        # this data frame only contains the 16 classes selected
        imagenet16class_df = self.image_info_df[
            self.image_info_df['class'].isin(imagenet_16_class_wnid_to_category.keys())]
        # self.image_info_df = imagenet16class_df
        self._width = imagenet16class_df['width']
        self._height = imagenet16class_df['height']
        self._path = imagenet16class_df['path']
        self._class = imagenet16class_df['class']
        self._id = imagenet16class_df['id']
        self._filename = imagenet16class_df['filename']

    # @property
    # def image_info_df(self):
    #
    #     data = {"id": self._id, "width": self._width, "height": self._height, "filename": self._filename,
    #             "path": self._path, "class": self._class}
    #     return pd.DataFrame(data=data)

    # def class_to_id(self, _class: str) -> int:

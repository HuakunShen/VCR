from __future__ import annotations

import os
import logging
import pandas as pd
from PIL import Image
from pathlib2 import Path
from typing import List, Union, Dict

from reliabilitycli.src.dataset_info import DatasetInfo


logger = logging.getLogger(__name__)


class ClassificationDatasetInfo(DatasetInfo):
    """
    This class assumes that the target dataset follows ImageFolder structure, i.e. each class is a folder and images
    are stored in the folder of their type
    """
    def __init__(self, data_root: Path, classes: List[str]=[]):
        super(ClassificationDatasetInfo, self).__init__(data_root)
        if (classes is None) or (len(classes) == 0):
            classes = list(os.listdir(self.data_root))
        self._class, self.classes = [], classes
        if not self.verify_root_validity():
            raise ValueError(
                f"The dataset root ({self.data_root}) doesn't contain folders for every class")

    def parse(self, ext: Union[str, List[str]] = 'png') -> ClassificationDatasetInfo:
        """
        (https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html#torchvision.datasets.ImageFolder)
        This parse function assumes that self.data_root is in ImageFolder structure
        Which is a typical folder structure for classification datasets
        Each folder in self.data_root is one of the class names and images are stored in the folder of their type

        If there is any special task this method doesn't satisfy, please override it in the child class

        :param ext: image extension(s) to search for, either single or multiple extensions is allowed
        :return: self
        """
        logger.info("Parse dataset")
        ext = ext if isinstance(ext, List) else [ext]
        for ext_ in ext:
            for idx, path in enumerate(self.data_root.glob(f'**/*.{ext_}')):
                self._id.append(idx)
                im = Image.open(str(path))
                width, height = im.size
                self._width.append(width)
                self._height.append(height)
                self._filename.append(path.name)
                self._path.append(str(path))
                self._class.append(path.parent.name)
                im.close()
        return self

    @property
    def image_info_df(self):
        """
        The returned dataframe will contain all columns as its parent class, with an extra class column
        dtype of class column will be string, storing the 10 classes of cifar10
        The reason to store the 10 classes is that, Cifar10 is stored as ImageFolder, which doesn't rely on a single
        file to store the labels, but folders. So we have to know it's label before saving the images.

        sample output data frame (ImageNet)
        +---+----+-------+--------+------------------------------+---------------------------------------------+-----------+
        |   | id | width | height |           filename           |                    path                     |   class   |
        +---+----+-------+--------+------------------------------+---------------------------------------------+-----------+
        | 0 | 0  |  500  |  375   | ILSVRC2012_val_00017472.JPEG | /val/n01440764/ILSVRC2012_val_00017472.JPEG | n01440764 |
        | 1 | 1  |  500  |  375   | ILSVRC2012_val_00025527.JPEG | /val/n01440764/ILSVRC2012_val_00025527.JPEG | n01440764 |
        | 2 | 2  |  500  |  375   | ILSVRC2012_val_00023559.JPEG | /val/n01440764/ILSVRC2012_val_00023559.JPEG | n01440764 |
        | 3 | 3  |  500  |  375   | ILSVRC2012_val_00009346.JPEG | /val/n01440764/ILSVRC2012_val_00009346.JPEG | n01440764 |
        | 4 | 4  |  500  |  375   | ILSVRC2012_val_00031094.JPEG | /val/n01440764/ILSVRC2012_val_00031094.JPEG | n01440764 |
        +---+----+-------+--------+------------------------------+---------------------------------------------+-----------+
        """
        data = {"id": self._id, "width": self._width, "height": self._height, "filename": self._filename,
                "path": self._path, "class": self._class}
        return pd.DataFrame(data=data)

    @property
    def filename_to_image_info_dict(self) -> Dict[str, Dict[str, Union[str, int]]]:
        """
        Map filename to image info dictionary
        We expect the images in original dataset to have unique filenames, otherwise this will not work
        The returned dictionary has key being filename, value being a dictionary containing the following columns
        - id
        - width
        - height
        - path
        - class
        :return: dictionary mapping a image filename in original dataset to its image info
        """
        return self.image_info_df.set_index("filename").to_dict("index")

    @property
    def image_info_dict(self) -> Dict[int, Dict[str, Union[str, int]]]:
        """
        Map image id to image info.
        The id here has nothing to do with images in the original dataset, just indices.
        The id is used
        The returned dictionary has key being id, value being a dictionary containing the following columns
        - filename
        - width
        - height
        - path
        - class
        :return: dictionary mapping a image filename in original dataset to its image info
        """
        return self.image_info_df.set_index('id').to_dict('index')

    def class_to_id(self, _class: str) -> int:
        """
        This method is for converting class to id
        Usually classes are in strings, but training or evaluation requires a number to compute.
        The class here must correspond to the class column in dataset info
        Datasets like ImageNet may be confusing, e.g. ImageNet have 2 types of classes, one english class, one wnid

        Here is how it's used
        image_info_dict = dataset_info.image_info_dict
        _class = image_info_dict[row['id']]['class']  # make sure _class corresponds to the class column here
        cls_idx = dataset_info.class_to_id(_class)  # this is the line this method will be called on
        """
        raise NotImplementedError

    def verify_root_validity(self) -> bool:
        """
        Verify that the data root is valid by checking whether the class directories all exist
        :return: valid or not
        """
        logger.info("Validating dataset, make sure all class folder exist")
        return all([d in self.classes for d in os.listdir(self.data_root)])

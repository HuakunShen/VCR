import os

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Callable, Tuple, Any

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from pathlib2 import Path
from torchvision.datasets import VOCDetection
from torchvision import transforms as T

from reliabilitycli.src.bootstrap.util import transform_image
from reliabilitycli.src.dataset_info import ClassificationDatasetInfo
from reliabilitycli.src.utils.transform import bootstrap_transform, toTensor
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

class DFDataset(Dataset):
    """
    Takes a pandas dataframe as input, the data frame contains all images and their transformation parameters
    The dataframe should contain the following columns
    - image_id
    - image_path: path to original image
    - transformation
    - transformation_parameter
    - vd_score
    """

    def __init__(self, df: pd.DataFrame, dataset_info: ClassificationDatasetInfo,
                 transform=torchvision.transforms.ToTensor(), target_transform=torch.tensor):
        self.df = df
        self.transform = transform
        self.dataset_info = dataset_info
        self.target_transform = target_transform
        self.image_info_dict = dataset_info.image_info_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx].to_dict()
        transformation_type = row['transformation']
        image_path = row['image_path']
        transformation_parameter = row['transformation_parameter']

        img = Image.open(image_path).convert("RGB")
        img_arr = np.array(img)
        img2, param = bootstrap_transform(img, transformation_type, transformation_parameter)
        img2 = img2.astype(np.uint8)
        # if transformation_parameter:
        #     img2, param = bootstrap_transform(img, transformation_type, transformation_parameter)
        # else:
        #     img2 = img_arr
        _class = self.image_info_dict[row['image_id']]['class']
        cls_idx = self.dataset_info.class_to_id(_class)
        label = self.target_transform(cls_idx)
        img3 = self.transform(img2)  # this transform is mainly converting np array image to tensor
        img4 = self.transform(img_arr)
        img.close()
        return {
            "original": img4,
            "transformed": img3,
            "label": label,
            "image_path": image_path,
            "transformation": transformation_type,
            "transformation_parameter": transformation_parameter,
            "vd_score": row['vd_score']
        }


class ImageFolderV2(Dataset):
    """
    A variant of ImageFolder provided by PyTorch
    https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html#torchvision.datasets.ImageFolder

    The PyTorch version doesn't allow empty class directory. It's possible in our experiment, some folders are empty.
    So I implemented a similar version that accepts empty class folder.
    A DataSet also needs to map an image to a label. So this class also requires a list of classes. The list should be
    in the correct order, they will be used to map index.
    """

    def __init__(self, root: Union[str, Path], classes: List[str], shuffle: bool = False,
                 transform=torchvision.transforms.ToTensor(), target_transform=torch.tensor):
        self.root = Path(root)
        self.classes = classes
        self.image_paths, self.labels = [], []
        self.class2label = {class_: idx for idx, class_ in enumerate(classes)}
        self.label2class = {value: key for key, value in self.class2label.items()}
        self.transform = transform
        self.target_transform = target_transform
        for class_ in self.classes:
            class_folder = self.root / class_
            assert class_folder.exists()
            for file in class_folder.iterdir():
                self.image_paths.append(str(file))
                self.labels.append(self.class2label[class_])
        self.df = pd.DataFrame(data={"path": self.image_paths, "label": self.labels})
        if shuffle:
            self.df = self.df.sample(frac=1)

    def __len__(self):
        """
        :return: number of images in dataset
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        retrieve an image
        :param idx: index to locate image in dataset
        :return: image tensor, label tensor and abs path to image
        """
        row = self.df.iloc[idx].to_dict()
        path, label = row['path'], self.target_transform(int(row['label']))
        im = Image.open(path).convert('RGB')
        img = self.transform(im)
        im.close()
        return img, label, path


class BootstrapDataset(Dataset):
    def __init__(self, root: Union[str, Path], classes: List[str], sample_size: int, transformation_type: str,
                 threshold: float, shuffle: bool = False, transform=torchvision.transforms.ToTensor(),
                 target_transform=torch.tensor):
        self.root = Path(root)
        self.transformation_type = transformation_type
        self.sample_size = sample_size
        self.threshold = threshold
        self.classes = classes
        self.image_paths, self.labels = [], []
        self.class2label = {class_: idx for idx, class_ in enumerate(classes)}
        self.label2class = {value: key for key, value in self.class2label.items()}
        self.transform = transform
        self.target_transform = target_transform
        for class_ in self.classes:
            class_folder = self.root / class_
            assert class_folder.exists()
            for file in class_folder.iterdir():
                self.image_paths.append(str(file))
                self.labels.append(self.class2label[class_])
        self.df = pd.DataFrame(data={"path": self.image_paths, "label": self.labels})
        if shuffle:
            self.df = self.df.sample(frac=1)
        self.bootstrap_df = None
        self.resample_bootstrap()

    def __len__(self):
        """
        :return: number of images in dataset
        """
        return self.sample_size

    def resample_bootstrap(self) -> pd.DataFrame:
        # sample_df = self.images_info_df.sample(n=self.sample_size, replace=False)
        self.bootstrap_df = self.df.sample(n=self.sample_size, replace=False)
        return self.bootstrap_df

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.bootstrap_df.iloc[idx].to_dict()
        path, label = row['path'], self.target_transform(int(row['label']))
        img = transform_image(self.transformation_type, path, self.threshold)
        img = self.transform(img['image'])
        return img, label, path


class CustomPascalVOCDataset(Dataset):
    """
    Custom dataset for loading Pascal VOC dataset.
    """

    def __init__(self, root: str, orig_root: str,
                 year: str = "2012",
                 image_set: str = "train",
                 download: bool = False,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None, ):
        super(CustomPascalVOCDataset, self).__init__()
        self.dataset = VOCDetection(orig_root, year, image_set, download, transform, target_transform, transforms)
        self.images_dir = Path(root).absolute() / 'images'
        self.annotations_dir = Path(root).absolute() / 'annotations'
        assert self.images_dir.exists()
        assert self.annotations_dir.exists()
        self.image_names = os.listdir(str(self.images_dir))
        self.image_paths = [self.images_dir / filename for filename in self.image_names]
        self.image_ids = [filename.split(".")[0] for filename in self.image_names]
        self.i2filename = {i: os.path.basename(path) for i, path in enumerate(self.dataset.images)}
        self.filename2i = {v: k for k, v in self.i2filename.items()}

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        The algorithm to match our custom datasets images to the original pascal voc images is very simple
        The original datasets takes in an index, the index is based on all images
        so I have to map custom index to original index

        1. Map custom index to the filename of selected image
        2. Map filename to original index
        """
        filename = self.image_names[index]
        original_index = self.filename2i[filename]
        return self.dataset[original_index]

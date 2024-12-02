import os
import cv2
import time
import pandas as pd
from tqdm import tqdm
from pathlib2 import Path
from typing import List, Dict
from torch.utils.data import DataLoader
from reliabilitycli.src.utils.path import clear_dir
from reliabilitycli.src.utils.transform import bootstrap_transform, get_image_based_on_transformation, toTensor
from reliabilitycli.src.dataset import ImageFolderV2


def link_original_image_folder(destination: Path, classes: List[str], image_path_to_class: Dict[str, str]):
    """
    Links original image to a new directory in image-folder-style structure.
    This is designed for image classification datasets.
    I need image path, and the corresponding classes
    :param destination: destination folder to link images
    :param classes: classes of dataset
    :param image_path_to_class: a dict mapping image path to class name (not index), e.g. {"/home/image.png": "car"}
    :return:
    """
    destination.mkdir(parents=True, exist_ok=True)
    clear_dir(destination)
    for class_ in classes:
        (destination / class_).mkdir(parents=True, exist_ok=True)
    for image_path, class_ in image_path_to_class.items():
        filename = os.path.basename(image_path)
        target_path = os.path.join(destination, class_, filename)
        os.link(image_path, target_path)


def save_image_folder(destination: Path, classes: List[str], bootstrap_df: pd.DataFrame, images_info_dict: Dict):
    """
    Save bootstrap images in ImageFolder style structure
    :param destination:
    :param classes: classes of dataset
    :param bootstrap_df:
    :param images_info_dict:
    """
    time.sleep(0.1)  # wait for previous log, other may collide with tqdm bar in this function
    destination.mkdir(parents=True, exist_ok=True)
    clear_dir(destination)
    pbar = tqdm(total=len(bootstrap_df), desc="Save")
    iteration_ids = bootstrap_df['iteration_id'].unique()
    for i in iteration_ids:
        iter_path = destination / f'iter{i}'
        iter_path.mkdir(parents=True, exist_ok=True)
        for class_ in classes:
            (iter_path / class_).mkdir(parents=True, exist_ok=True)
        bootstrap_iter_df = bootstrap_df[bootstrap_df['iteration_id'] == i]
        image_ids = bootstrap_iter_df['image_id'].unique().tolist()
        assert len(image_ids) == len(bootstrap_iter_df['image_id'])  # image shouldn't repeat
        for index, row in bootstrap_iter_df.iterrows():
            image_info_dict = images_info_dict[row['image_id']]
            img = get_image_based_on_transformation(row['transformation'], image_info_dict['path'])
            img2, param_index = bootstrap_transform(img, row['transformation'],
                                                    row['transformation_parameter'])
            output_path = iter_path / image_info_dict['class'] / image_info_dict['filename']
            cv2.imwrite(str(output_path), img2)
            pbar.update(1)


def validate_image_folder(destination: Path, classes: List[str], tqdm_total: int, batch_size: int = 1,
                          loader_transform=toTensor) -> None:
    """
    Validate that a given folder (dataset) can be loaded with ImageFolderV2.
    :param destination: path to dataset folder
    :param classes: expected classes of the dataset
    :param tqdm_total: total number of expected images, used for progress bar.
    :param batch_size: batch size parameter for data loader
    :param loader_transform: If images have different dimension, you need to provide reshaping transformation
    :return: None
    """
    time.sleep(0.1)  # wait for previous log, other may collide with tqdm bar in this function
    pbar = tqdm(total=tqdm_total, desc="Validate")
    for path in destination.iterdir():
        dataset = ImageFolderV2(path, classes=classes, transform=loader_transform)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for _ in dataloader:
            pbar.update(batch_size)

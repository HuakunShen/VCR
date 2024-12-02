from __future__ import annotations

import time
import pickle
import logging
import pandas as pd
from tqdm import tqdm
from typing import Union
from pathlib2 import Path
from skimage.color import rgb2gray
from abc import ABC, abstractmethod
from skimage.metrics import structural_similarity as ssim

from reliabilitycli.src.dataset_info import DatasetInfo, ClassificationDatasetInfo
from reliabilitycli.src.utils.transform import get_image_based_on_transformation, bootstrap_transform
from reliabilitycli.src.workspace import Workspace
from reliabilitycli.src.utils.bootstrap import save_image_folder, validate_image_folder, link_original_image_folder
from reliabilitycli.src.utils.transform import toTensor

logger = logging.getLogger(__name__)

class Bootstrapper(ABC):
    """
    Bootstrapper Abstract Class
    The order to run is run, save, validate
    e.g. bootstrapper.run().save().validate()
    """

    def __init__(self, num_sample_iter: int, sample_size: int,
                 workspace: Workspace, dataset_info: DatasetInfo, transformation_type: str,
                 threshold: float):
        self.num_sample_iter = num_sample_iter
        self.sample_size = sample_size
        self.dataset_info = dataset_info
        self.images_info_df = self.dataset_info.image_info_df
        self.transformation_type = transformation_type
        self.threshold = threshold
        self.workspace = workspace
        self.data = None
        self.bootstrap_df = None

    @abstractmethod
    def save(self) -> Bootstrapper:
        """
        Save transformed images to bootstrap output folder, this should be run after run() has been run
        :return: self, following Chain-of-responsibility pattern
        """
        raise NotImplementedError

    @abstractmethod
    def validate(self, batch_size: int = 1) -> Bootstrapper:
        """
        Validate generated images (dataset subsets) can be properly loaded
        :param batch_size: batch size to use for PyTorch data loader
        :return: self, following Chain-of-responsibility pattern
        """
        raise NotImplementedError

    @property
    def logger(self) -> logging.Logger:
        """
        :return: logger for the project retrieved from Workspace singleton object
        """
        return Workspace.instance().get_logger()

    def run(self) -> Bootstrapper:
        """
        run bootstrapping process, produce an instance variable called bootstrap_df
        bootstrap_df is a pandas DataFrame with the following columns
        - iteration_id
        - within_iter_id
        - image_id
        - transformation_type
        - transformation_parameter
        - vd_score
        :return: self, following Chain-of-responsibility pattern
        """
        logger.info("Generating Bootstrap Information")
        time.sleep(0.1)  # wait for previous log, other may collide with tqdm bar in this function
        progress_bar = tqdm(total=self.num_sample_iter * self.sample_size, desc="Bootstrapping")
        time.sleep(0.1)
        bootstrap_decisions = []
        for i in range(1, self.num_sample_iter + 1):
            sample_df = self.images_info_df.sample(n=self.sample_size, replace=False)
            within_iter_count = 1
            image_ids_selected = set()
            for index, cur_row in sample_df.iterrows():
                if cur_row['id'] in image_ids_selected:
                    continue
                image_path = cur_row['path']
                img = get_image_based_on_transformation(self.transformation_type, image_path)
                while True:
                    img2, param_index = bootstrap_transform(img, self.transformation_type)
                    gray_1, gray_2 = rgb2gray(img), rgb2gray(img2)
                    ssim_noise = ssim(gray_1, gray_2,
                                      data_range=gray_1.max() - gray_2.min())
                    if 1 - ssim_noise < self.threshold:
                        bootstrap_decisions.append({
                            'iteration_id': i,
                            'within_iter_id': within_iter_count,
                            'image_id': cur_row['id'],
                            'transformation': self.transformation_type,
                            'transformation_parameter': param_index,
                            'vd_score': 1 - ssim_noise
                        })
                        image_ids_selected.add(cur_row['id'])
                        break
                within_iter_count += 1
                progress_bar.update(n=1)
        self.bootstrap_df = pd.DataFrame(data=bootstrap_decisions)
        return self
    
    def dump(self):
        """Save bootstrapper object in pickle format as cache"""
        with open(self.workspace.get_workspace_path() / 'bootstrapper.pickle', 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: Union[Path, str]) -> Bootstrapper:
        with open(str(path), 'rb') as f:
            return pickle.load(f)
    
    def __str__(self) -> str:
        val = f"""
        Bootstrapper
        - num_sample_iter: {self.num_sample_iter}
        - sample_size: {self.sample_size}
        - transformation: {self.transformation_type}
        - threshold: {self.threshold}
        - workspace: {self.workspace}
        """
        return val


class ClassificationBootstrapper(Bootstrapper):
    """
    Bootstrapper designed for Classification datasets that supports ImageFolder structure
    Should be compatible with all datasets as long as it can be read with PyTorch ImageFolder
    """

    def __init__(self, num_sample_iter: int, sample_size: int,
                 workspace: Workspace,
                 dataset_info: ClassificationDatasetInfo, transformation_type: str, threshold: float):
        super(ClassificationBootstrapper).__init__(num_sample_iter, sample_size, workspace, dataset_info, transformation_type, threshold)
        # unnecessary, for intellisense to work since type changed to ImagenetDatasetInfo
        self.dataset_info = dataset_info

    def save(self) -> ClassificationBootstrapper:
        logger.info("Saving Bootstrap Results in ImageFolder Structure")
        save_image_folder(self.workspace.bootstrap_path, self.dataset_info.classes, self.bootstrap_df,
                          self.dataset_info.image_info_dict)
        image_info_dict = self.dataset_info.image_info_dict
        image_path_to_class = {image_info_dict[image_id]['path']: image_info_dict[image_id]['class'] for image_id in
                               self.bootstrap_df['image_id'].unique()}
        link_original_image_folder(self.workspace.original_path, self.dataset_info.classes, image_path_to_class)
        return self

    def validate(self, batch_size: int = 1) -> ClassificationBootstrapper:
        logger.info("Validating Generated Bootstrap Dataset Directories")
        validate_image_folder(self.workspace.bootstrap_path, classes=self.dataset_info.classes,
                              tqdm_total=self.num_sample_iter * self.sample_size,
                              batch_size=batch_size, loader_transform=toTensor)
        return self

from typing import Dict

from pathlib2 import Path

import cv2
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
# import matplotlib.pyplot as plt
# from skimage.color import rgb2gray
# from skimage.metrics import structural_similarity as ssim
# from sewar.full_ref import vifp
import multiprocessing as mp

from reliabilitycli.src.utils.bootstrap import link_original_image_folder
from reliabilitycli.src.utils.log import get_custom_logger_formatter
from reliabilitycli.src.utils.path import clear_dir
from reliabilitycli.src.utils.transform import get_image_based_on_transformation, bootstrap_transform
from reliabilitycli.src.dataset_info import DatasetInfo, ClassificationDatasetInfo
from reliabilitycli.src.workspace import Workspace
from reliabilitycli.src.constants.imagenet_16_class_map import imagenet_16_class_wnid_to_category
from reliabilitycli.src.utils.vif_utils import vif
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# from vif——utils import vif
# # vif needs original and transformed images both to be grayscale, convert if not
# transformed_img_g = transformed_img.astype('float32')
# if len(transformed_img_g.shape) > 2:
#  orig_img_g = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
#  transformed_img_g = cv2.cvtColor(transformed_img_g, cv2.COLOR_RGB2GRAY)
# IQA_score = vif(orig_img_g, transformed_img_g) # calculate vif value

logger = logging.getLogger(__name__)
logger.addHandler(get_custom_logger_formatter())


def single_image_transform(data: Dict):
    image_path = data['image_path']
    transformation_type = data['transformation']
    image_id = data['image_id']
    idx = data['idx']
    img = Image.open(image_path).convert("RGB")
    img_arr = np.asarray(img)
    img2, param_index = bootstrap_transform(img, transformation_type)
    img2 = img2.astype(np.uint8)
    if len(img2.shape) > 2:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    iqa_score = vif(img_arr, img2)
    # noise = vifp(img_arr, img2)
    return {
        'id': idx,
        'image_id': image_id,
        'image_path': image_path,
        'transformation': transformation_type,
        'transformation_parameter': param_index,
        'vd_score': 1 - iqa_score
    }


class ImageGenerator:
    def __init__(self, workspace: Workspace, sample_size: int, transformation_type: str,
                 dataset_info: ClassificationDatasetInfo):
        self.sample_size = sample_size
        self.transformation_type = transformation_type
        self.workspace = workspace
        self.dataset_info = dataset_info
        self.images_info_df = self.dataset_info.image_info_df
        self.df: pd.DataFrame = None

    def run(self):
        sample_df = self.images_info_df.sample(n=self.sample_size, replace=True)
        # print(sample_df)
        # image_ids_selected = set()
        # sample_results = []
        # pbar = tqdm(sample_df.iterrows(), total=len(sample_df), desc='Image Generator Running')
        sample_df.reset_index(inplace=True)
        data = [
            {"image_path": cur_row['path'], "transformation": self.transformation_type, "image_id": cur_row['id'],
             "idx": index} for index, cur_row in sample_df.iterrows()]
        # with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        #     sample_results = list(
        #         tqdm(executor.map(single_image_transform, data), total=len(data),
        #              desc='Image Generator Running'))

        with mp.Pool(mp.cpu_count()) as pool:
            sample_results = list(
                tqdm(pool.imap(single_image_transform, data), total=len(data), desc='Image Generator'))
        #     print("finished")
        # for index, cur_row in pbar:
        #     if cur_row['id'] in image_ids_selected:
        #         continue
        #     image_path = cur_row['path']
        #     # img = get_image_based_on_transformation(self.transformation_type, image_path)
        #     img = Image.open(image_path).convert("RGB")
        #     img_arr = np.asarray(img)
        #     img2, param_index = bootstrap_transform(img, self.transformation_type)
        #     img2 = img2.astype(np.uint8)
        #     noise = vifp(img_arr, img2)
        #     pbar.set_postfix({"noise": noise})
        #     sample_results.append({
        #         'id': index,
        #         'image_id': cur_row['id'],
        #         'image_path': image_path,
        #         'transformation_type': self.transformation_type,
        #         'transformation_parameter': param_index,
        #         'vd_score': 1 - noise
        #     })
        #     image_ids_selected.add(cur_row['id'])
        self.df = pd.DataFrame(data=sample_results)
        return self

    @property
    def sample_df_save_path(self) -> Path:
        return self.workspace.sample_df_save_path

    def save_df(self):
        assert self.df is not None, "dataframe not generated yet, call .run() first"
        self.df.to_csv(self.sample_df_save_path)
        return self

    def save(self):
        print(self.df)
        if self.df is None:
            raise ValueError("Run the generator first")
        self.workspace.bootstrap_path.mkdir(parents=True, exist_ok=True)
        clear_dir(self.workspace.bootstrap_path)
        pbar = tqdm(total=len(self.df), desc="Save")
        for class_ in self.dataset_info.classes:
            (self.workspace.bootstrap_path /
             class_).mkdir(parents=True, exist_ok=True)
        image_ids = self.df['image_id'].unique().tolist()
        assert len(image_ids) == len(
            self.df['image_id'])  # image shouldn't repeat
        for i, row in self.df.iterrows():
            image_info_dict = self.dataset_info.image_info_dict[row['image_id']]
            img = get_image_based_on_transformation(
                row['transformation'], image_info_dict['path'])
            img2, param_index = bootstrap_transform(img, row['transformation'],
                                                    row['transformation_parameter'])
            img2 = img2.astype(np.uint8)

            output_path = self.workspace.bootstrap_path / \
                          image_info_dict['class'] / image_info_dict['filename']
            cv2.imwrite(str(output_path), img2)
            pbar.update(1)
        image_path_to_class = {
            self.dataset_info.image_info_dict[image_id]['path']: self.dataset_info.image_info_dict[image_id]['class']
            for image_id in
            self.df['image_id'].unique()}
        link_original_image_folder(self.workspace.original_path,
                                   classes=self.dataset_info.classes,
                                   image_path_to_class=image_path_to_class)
        return self

# This class is discarded, using custom dataset info is a better choice (Imagenet16ClassDatasetInfo)
# class ImagetNet16ImageGenerator(ImageGenerator):
#     def __init__(self, workspace: Workspace, sample_size: int, transformation_type: str, dataset_info: ClassificationDatasetInfo):
#         super().__init__(workspace, sample_size, transformation_type, dataset_info)
#
#     def run(self):
#         imagenet16class_df = self.images_info_df[self.images_info_df['class'].isin(imagenet_16_class_wnid_to_category.keys())]
#         sample_df = imagenet16class_df.sample(n=self.sample_size, replace=True)
#         sample_df.reset_index(inplace=True)
#         data = [
#             {"image_path": cur_row['path'], "transformation": self.transformation_type, "image_id": cur_row['id'],
#              "idx": index} for index, cur_row in sample_df.iterrows()]
#
#         with mp.Pool(mp.cpu_count()) as pool:
#             sample_results = list(
#                 tqdm(pool.imap(single_image_transform, data), total=len(data), desc='Image Generator'))
#         self.df = pd.DataFrame(data=sample_results)
#         return self
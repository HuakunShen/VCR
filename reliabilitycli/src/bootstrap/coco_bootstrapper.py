from typing import Union

import cv2
from pathlib2 import Path

from reliabilitycli.src.bootstrap.bootstrapper import Bootstrapper
from reliabilitycli.src.dataset_info.coco_dataset_info import CocoDatasetInfo
from reliabilitycli.src.utils.path import clear_dir
from reliabilitycli.src.utils.transform import get_image_based_on_transformation, bootstrap_transform


class CocoBootstrapper(Bootstrapper):
    def __init__(
            self, num_sample_iter: int, sample_size: int,
            destination: Union[str, Path],
            dataset_info: CocoDatasetInfo, transformation_type: str, threshold: float):
        super().__init__(num_sample_iter, sample_size, destination, dataset_info, transformation_type, threshold)
        self.dataset_info = dataset_info

    def save(self):
        images_info_df = self.dataset_info.image_info_df
        root = Path('./workspace').absolute()
        bootstrap_path = root / 'bootstrap'
        bootstrap_path.mkdir(parents=True, exist_ok=True)
        clear_dir(bootstrap_path)
        iteration_ids = self.bootstrap_df['iteration_id'].unique()
        images_info_dict = images_info_df.set_index('id').to_dict('index')
        for i in iteration_ids:
            iter_path = bootstrap_path / f'iter{i}'
            iter_path.mkdir(parents=True, exist_ok=True)
            images_dir = iter_path / 'images'
            images_dir.mkdir(parents=True, exist_ok=True)
            bootstrap_iter_df = self.bootstrap_df[self.bootstrap_df['iteration_id'] == i]
            image_ids = bootstrap_iter_df['image_id'].unique().tolist()
            assert len(image_ids) == len(bootstrap_iter_df['image_id'])  # image shouldn't repeat

            for index, row in bootstrap_iter_df.iterrows():
                image_info_dict = images_info_dict[row['image_id']]
                img = get_image_based_on_transformation(row['transformation_type'], image_info_dict['path'])
                img2, param_index = bootstrap_transform(img, row['transformation_type'],
                                                        row['transformation_parameter'])
                output_path = images_dir / image_info_dict['filename']
                cv2.imwrite(str(output_path), img2)
            annotation_json = iter_path / 'annotations.json'
            with open(annotation_json, 'w') as f:
                f.write(self.dataset_info.subset_to_json(image_ids))

import io
import json
from typing import List, Dict

import pandas as pd
from pathlib2 import Path

from reliabilitycli.src.dataset_info.dataset_info import DatasetInfo


class CocoDatasetInfo(DatasetInfo):
    def __init__(self, annotations_json: Path, data_root: Path):
        super(CocoDatasetInfo, self).__init__(Path(data_root))
        self.annotations_json = annotations_json
        with open(str(annotations_json)) as f:
            self.data = json.load(f)
        self.images_dict = {image['id']: image for image in self.data['images']}
        self.annotations_dict = {image_id: [] for image_id in self.images_dict.keys()}

        for annotation in self.data['annotations']:
            self.annotations_dict[annotation['image_id']].append(annotation)

    def get_full_image_info_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.data['images'])

    @property
    def image_info_df(self) -> pd.DataFrame:
        """
        - id
        - filename
        - path
        - width
        - height

        :return: [description]
        :rtype: pd.DataFrame
        """
        full_df = self.get_full_image_info_df()
        subset_df = full_df[['id', 'file_name', 'width', 'height']].copy()  # df containing only image info, no object
        subset_df.loc[:, 'path'] = subset_df['file_name'].apply(lambda filename: str(self.data_root / filename))
        subset_df = subset_df.rename(columns={'file_name': 'filename'}, errors='raise')
        for path in subset_df['path']:
            assert Path(path).exists()
        return subset_df

    def subset_to_json_dict(self, image_ids: List[int], stream: io.TextIOWrapper = None) -> Dict:
        data = {'images': [self.images_dict[image_id] for image_id in image_ids]}
        data.update({'annotations': [obj for image_id in image_ids for obj in self.annotations_dict[image_id]]})
        data.update({'categories': self.data['categories']})
        return data

    def subset_to_json(self, image_ids: List[int], stream: io.TextIOWrapper = None) -> str:
        data = self.subset_to_json_dict(image_ids, stream)
        json_str = json.dumps(data, indent=2)
        if stream is not None:
            stream.write(json_str)
        return json_str
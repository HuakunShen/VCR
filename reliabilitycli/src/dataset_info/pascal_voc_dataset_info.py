import os
from xml.etree import ElementTree as ET

import pandas as pd
from pathlib2 import Path

from reliabilitycli.src.dataset_info.dataset_info import DatasetInfo


class PascalVOCDatasetInfo(DatasetInfo):
    def __init__(self, root: Path, image_set: str = "val"):
        super(PascalVOCDatasetInfo, self).__init__()
        self.root = Path(root)
        self.image_root = self.root / "VOCdevkit" / "VOC2007" / "JPEGImages"
        self.annotation_root = self.root / "VOCdevkit" / "VOC2007" / "Annotations"
        self.image_set = image_set
        assert self.image_set in ["train", "val", "trainval"]
        self.image_list_file = self.root / "VOCdevkit" / "VOC2007" / "ImageSets" / "Main" / f"{self.image_set}.txt"
        with open(str(self.image_list_file), "r") as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        self.image_filenames = [f"{id_}.jpg" for id_ in self.image_ids]
        if not self.verify_root_validity():
            raise ValueError("Annotation and Image Root Incompatible, annotation must be a superset of image files.")

    def verify_root_validity(self) -> bool:
        """
        whether the image_root and annotation_root used to initialize this datasets info object is valid, or compatible
        image files is what we look at, and annotation files can be a super set of image files
        :return: is valid
        """
        image_filenames = os.listdir(str(self.image_root))
        # this is different from self.image_ids, this is what's on disk, self.image_ids depends on image_set
        image_ids = set([filename.split(".")[0] for filename in image_filenames])
        annotation_filenames = os.listdir(str(self.annotation_root))
        annotation_ids = set([filename.split(".")[0] for filename in annotation_filenames])
        return len(image_ids & annotation_ids & set(self.image_ids)) >= len(self.image_ids)

    @property
    def image_info_df(self) -> pd.DataFrame:
        """

        :return: image info dataframe
        :rtype: pd.DataFrame
        """
        image_ids = set([filename.split(".")[0] for filename in self.image_filenames])
        data = {"id": [], "width": [], "height": [], "filename": [], "path": []}
        for id_ in image_ids:
            tree = ET.parse(str(self.annotation_root / f"{id_}.xml"))
            root = tree.getroot()
            filename = f"{id_}.jpg"
            data["id"].append(id_)
            data["width"].append(int(root.findall("size/width")[0].text))
            data["height"].append(int(root.findall("size/height")[0].text))
            data["filename"].append(filename)
            data["path"].append(str(self.image_root / filename))
        return pd.DataFrame(data=data)
import os
import sys
import torch
import shutil
import logging
import pathlib2
import argparse
import torchvision
import pandas as pd
from tqdm import tqdm
from typing import Union
import torch.multiprocessing

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(filename)s | %(lineno)d | %(message)s',
                              '%m-%d-%Y %H:%M:%S')
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

torch.multiprocessing.set_sharing_strategy('file_system')
data_dir = pathlib2.Path().home().absolute() / 'datasets'

if not data_dir.exists():
    data_dir.mkdir(parents=True, exist_ok=True)
toPIL = torchvision.transforms.ToPILImage()


def save(dataset_root: Union[pathlib2.Path, str], output_dir: Union[pathlib2.Path, str], train: bool):
    output_dir = pathlib2.Path(output_dir).absolute()
    dataset_root = pathlib2.Path(dataset_root).absolute()
    logger.info(f"Remove target output directory {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = torchvision.datasets.CIFAR10(root=dataset_root, train=train, download=True,
                                           transform=torchvision.transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    logger.info(f"Saving images to {output_dir}")
    pbar = tqdm(total=len(dataset))
    classes = ["airplanes", "cars", "birds", "cats", "deer", "dogs", "frogs", "horses", "ships", "trucks"]
    for class_ in classes:
        os.makedirs(os.path.join(output_dir, class_), exist_ok=True)
    image_paths, label_ints, labels = [], [], []
    for i, (data, label) in enumerate(dataloader):
        img = toPIL(data[0])
        filename = f'{i}.png'
        image_path = os.path.join(output_dir, classes[int(label[0])], filename)
        img.save(image_path)
        image_paths.append(image_path)
        label_ints.append(int(label[0]))
        labels.append(classes[int(label[0])])
        img.close()
        pbar.update()
    label_df = pd.DataFrame({
        "image_path": image_paths,
        "label_int": label_ints,
        "label": labels
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Cifar10 Preprocessor")
    parser.add_argument("--data_dir", default=str(data_dir), type=str, help="root of datasets")
    args = parser.parse_args()
    data_dir_ = pathlib2.Path(args.data_dir).absolute() / 'cifar10'
    save(data_dir_, data_dir_ / "cifar10_pytorch" / "val", False)
    save(data_dir_, data_dir_ / "cifar10_pytorch" / "train", True)
    logger.info("Finished")

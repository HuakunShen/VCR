import logging
import os
import torch
import random
import argparse
import pathlib2
import numpy as np
import torchvision
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib2 import Path

from reliabilitycli.commands.eval import setup_eval_parser, val, evaluate
from reliabilitycli.commands.exp import experiment
from reliabilitycli.commands.bootstrap import bootstrap, setup_bootstrap_parser
from reliabilitycli.commands.sample import sample, setup_sample_parser
from reliabilitycli.commands.analyze import setup_analyze_parser, analyze
from reliabilitycli.commands.train import retrain
from reliabilitycli.src.bootstrap.bootstrapper import Bootstrapper
from reliabilitycli.src.constants import CIFAR10, IMAGENET
from reliabilitycli.src.utils.log import CustomFormatter, get_custom_logger_formatter

from reliabilitycli.src.workspace import Workspace, Config
from reliabilitycli.datasets.imagenet import ImagenetDatasetInfo
from reliabilitycli.datasets.cifar10 import Cifar10DatasetInfo
from reliabilitycli.src.dataset import ImageFolderV2
from reliabilitycli.src.constants.transformation import IMAGENET_DEFAULT_TRANSFORMATION
from reliabilitycli.src.utils.visualize import table2str
from reliabilitycli.src.utils.model import load_custom_model, get_model

toTensor = torchvision.transforms.ToTensor()

PROJECT_NAME = "cli-tool"

home = pathlib2.Path.home()
datasets_folder = home / 'datasets'
random.seed(10)
np.random.seed(10)
torch.manual_seed(10)


def main():
    # python main.py -w ./workspace run
    logger.info("main")
    w = Workspace.instance()
    w.setup_workspace()
    w.config.verify_config()
    bootstrapper = Bootstrapper.load(
        w.get_workspace_path() / 'bootstrapper.pickle')
    if w.config.config['dataset_name'] == IMAGENET:
        dataset_info = ImagenetDatasetInfo(data_root=w.config.dataset_dir / 'ImageNet' / 'val',
                                           original_data_root=w.config.dataset_dir / 'ImageNet',
                                           original_split='val')
    elif w.config.config['dataset_name'] == CIFAR10:
        dataset_info = Cifar10DatasetInfo(
            data_root=w.config.dataset_dir / 'cifar10' / 'cifar10_pytorch')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 5

    if w.config.config['model_name'] == "custom":
        model = load_custom_model().to(device).eval()
    else:
        model = get_model(w.config.config['model_name'])(pretrained=True).to(device).eval()

    label_list, pred_list, filenames = [], [], []
    pbar = tqdm(desc="Eval Bootstrap Images",
                total=bootstrapper.num_sample_iter * bootstrapper.sample_size)
    for path in w.bootstrap_path.iterdir():
        dataset = ImageFolderV2(
            path, classes=dataset_info.classes, transform=IMAGENET_DEFAULT_TRANSFORMATION)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for inputs, labels, image_paths in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            for i in range(len(image_paths)):
                iter_dir, class_, filename = image_paths[i].split('/')[-3:]
                filenames.append(filename)
            pred = torch.argmax(outputs, dim=1)
            label_list.extend(labels.tolist())
            pred_list.extend(pred.cpu().tolist())
            pbar.update(batch_size)
    bootstrap_res_df = pd.DataFrame(
        data={"label": label_list, "pred": pred_list, "filename": filenames})
    # evaluate original images
    # maps filename to image info like image id, class, path
    filename2image_info = dataset_info.filename_to_image_info_dict
    label_list, pred_list, filenames = [], [], []
    dataset = ImageFolderV2(w.original_path, classes=dataset_info.classes,
                            transform=IMAGENET_DEFAULT_TRANSFORMATION)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    pbar = tqdm(desc="Eval Original Images", total=len(dataset))
    for inputs, labels, image_paths in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        for i in range(len(image_paths)):
            iter_dir, class_, filename = image_paths[i].split('/')[-3:]
            filenames.append(filename)
        pred = torch.argmax(outputs, dim=1)
        label_list.extend(labels.tolist())
        pred_list.extend(pred.cpu().tolist())
        pbar.update(batch_size)
    original_res_df = pd.DataFrame(
        data={"filename": filenames, "label": label_list, "pred": pred_list})
    original_filename2dict = original_res_df.set_index(
        "filename").to_dict("index")
    bootstrap_res_df['orig_pred'] = bootstrap_res_df.apply(lambda row: original_filename2dict[row['filename']]['pred'],
                                                           axis=1)
    bootstrap_by_image_id = bootstrapper.bootstrap_df.set_index("image_id")
    bootstrap_res_df['image_id'] = bootstrap_res_df.apply(lambda row: filename2image_info[row['filename']]['id'],
                                                          axis=1)
    bootstrap_res_df['vd_score'] = bootstrap_res_df.apply(
        lambda row: bootstrap_by_image_id.loc[row['image_id']]['vd_score'], axis=1)
    print(table2str(bootstrap_res_df))


def estimate():
    w = Workspace.instance()
    w.setup_workspace()
    w.config.verify_config()
    logger.info("estimate")


def test():
    w = Workspace.instance()
    w.setup_workspace()
    w._config.verify_config()
    logger.info("test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(PROJECT_NAME)
    parser.add_argument('-w', '--workspace',
                        default='./workspace', help='Workspace Path')
    subparsers = parser.add_subparsers(help='sub-command help', dest='command')
    gen_config_parser = subparsers.add_parser(
        'gen-config', help='Generate Configuration')
    gen_config_parser.add_argument(
        '-o', '--overwrite', action='store_true', help='Overwrite Configuration File')
    run_parser = subparsers.add_parser('run', help='run help')
    exp_parser = subparsers.add_parser('experiment', help='experiment help')

    estimate_parser = subparsers.add_parser('estimate', help='estimate help')
    test_parser = subparsers.add_parser('test', help='test help')
    retrain_parser = subparsers.add_parser('retrain', help='retrain help')
    eval_parser = subparsers.add_parser('eval', help='eval help')
    setup_eval_parser(eval_parser)
    # setup_bootstrap_parser(subparsers)
    sample_parser = subparsers.add_parser('sample', help='Sample Images')
    setup_sample_parser(sample_parser)
    analyze_parser = subparsers.add_parser('analyze', help='Analyzer help')
    setup_analyze_parser(analyze_parser)

    args = parser.parse_args()
    w = Workspace.instance()
    w.initialize(Path(args.workspace).absolute(), PROJECT_NAME, args=args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = w.get_logger()
    # logger.addHandler(get_custom_logger_formatter())
    logger.warning("debug logger")
    if args.command == 'gen-config':
        logger.info("Generate Configuration")
        config = Config()
        w.set_config(config)
    elif args.command == 'bootstrap':
        logger.info("Start Bootstrap")
        bootstrap()
    elif args.command == 'run':
        logger.info("Start Running")
        main()
    elif args.command == 'sample':
        logger.info('Start Sampling')
        sample()
    elif args.command == 'eval':
        logger.info("eval_bootstrap_dataset")
        evaluate(w, device)
    elif args.command == 'experiment':
        logger.info("Start Experiment")
        experiment()
    elif args.command == 'estimate':
        logger.info("Start Estimation")
        estimate()
    elif args.command == 'test':
        logger.info("Start Testing")
        test()
    elif args.command == 'retrain':
        logger.info("Start Retraining")
        retrain()
    elif args.command == 'analyze':
        logger.info("Start Analyzing")
        # analyze(w, w.get_workspace_path() / args.eval_csv)
        analyze(w, w.get_workspace_path() / args.eval_csv, w.get_workspace_path() / 'human_results.csv')
    else:
        logger.error(f"Invalid Command {args.command}")

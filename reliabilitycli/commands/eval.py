from typing import List

import torchvision.transforms
from PIL import Image
import tabulate
import torch
import pandas as pd
from torchvision.models import resnet50
from tqdm import tqdm
from pathlib2 import Path
import multiprocessing as mp
# from torchvision.models import get_model
from torch.utils.data import DataLoader
from dataclasses import dataclass
from robustbench import load_model
import torchvision.transforms as transforms

from reliabilitycli.src.constants import IMAGENET_DEFAULT_TRANSFORMATION, TENSOR_TRANSFORMATION
from reliabilitycli.src.dataset import ImageFolderV2, DFDataset
from reliabilitycli.src.dataset_info.factory import dataset_info_factory
from reliabilitycli.src.utils.model import load_custom_model, get_model
from reliabilitycli.src.workspace import Workspace
from robustbench.loaders import CustomImageFolder
from robustbench.utils import clean_accuracy, load_model
from argparse import ArgumentParser


def setup_eval_parser(eval_parser: ArgumentParser):
    eval_parser.add_argument('-o', '--output', type=str, default='eval_results.csv',
                             help='Output filename of evaluation results csv file (just filename, not path, '
                                  'file will always be saved in workspace folder)')
    eval_parser.add_argument('-s', '--sample-csv-output', type=str,
                               help='Output Filename. (not path, file will be stored in workspace)')
    # eval_parser.add_argument(
    #     '--num_sample_iter', type=int, required=True, help='Number of sample iterations')
    # eval_parser.add_argument(
    #     '--sample_size', type=int, required=True, help='Bootstrap Sample Size')


@dataclass
class EvalStats:
    n: int
    n_trans_vs_orig_correct: int
    n_transformed_correct: int
    n_orig_correct: int

    @property
    def original_accuracy(self):
        return self.n_orig_correct / self.n

    @property
    def original_accuracy_str(self):
        return f"{round(self.original_accuracy * 100, 2)}%"

    @property
    def transformed_accuracy(self):
        return self.n_transformed_correct / self.n

    @property
    def transformed_accuracy_str(self):
        return f"{round(self.transformed_accuracy * 100, 2)}%"


def get_eval_stats(df: pd.DataFrame):
    n = len(df)
    n_trans_vs_orig_correct = (df['transf_pred'] == df['orig_pred']).sum()
    n_transformed_correct = (df['transf_pred'] == df['label']).sum()
    n_orig_correct = (df['label'] == df['orig_pred']).sum()
    eval_stats = EvalStats(n, n_trans_vs_orig_correct, n_transformed_correct, n_orig_correct)
    return eval_stats


toTensorCompose = transforms.Compose([transforms.ToTensor()])


def vis_tensor(data: torch.Tensor):
    # img = data[0].cpu().numpy()
    toPIL = torchvision.transforms.ToPILImage()
    img = toPIL(data)
    return img


def evaluate(w: Workspace, device: torch.device):
    print(w.get_args().output)
    w.setup_workspace()
    w.config.verify_config()
    if not w.sample_df_save_path.exists():
        raise ValueError(f"Path Doesn't Exist: {w.sample_df_save_path}")
    # if w.config.config['model'] == "custom":
    #     model = load_custom_model().to(device).eval()
    # else:
    #     model = get_model(w.config.config['model'])(weights='DEFAULT').to(device).eval()
    # Rebuffi2021Fixing_70_16_cutmix_extra
    # model = resnet50(pretrained=True).to(device).eval()
    # model = load_model('Salman2020Do_R18', dataset='imagenet', threat_model='Linf').to(device).eval()
    # model = load_model('Standard_R50', dataset='imagenet', threat_model='Linf').to(device).eval()
    # model = load_model('Debenedetti2022Light_XCiT-L12', dataset='imagenet', threat_model='Linf').to(device).eval()
    # model = load_model('Debenedetti2022Light_XCiT-L12', dataset='imagenet', threat_model='Linf').to(device).eval()
    # model = load_model('Hendrycks2020Many', dataset='imagenet', threat_model='corruptions').to(device).eval()
    # model = load_model('Standard_R50', dataset='imagenet', threat_model='corruptions').to(device).eval()
    # model = load_model('Engstrom2019Robustness', dataset='imagenet', threat_model='Linf').to(device).eval()
    models = w.load_models()

    df = pd.read_csv(w.sample_df_save_path, index_col=0)
    dataset_info = dataset_info_factory(w.config.dataset_name, w.config.dataset_dir)
    dst = DFDataset(df, dataset_info, transform=IMAGENET_DEFAULT_TRANSFORMATION)
    dataloader = DataLoader(dst, batch_size=100, num_workers=10)
    # x_test, y_test, paths =
    # original, transformed, labels, image_paths = next(iter(dataloader))
    # acc = clean_accuracy(model, transformed.to(device), labels.to(device), device=device)
    # print(f"acc: {acc}")
    pbar = tqdm(dataloader, desc='Evaluation', total=len(dst) * len(models))
    data = []
    for model_name, model in models:
        model.to(device).eval()
        # pbar = tqdm(dataloader, desc='Evaluation', total=len(dst))
        total_so_far, orig_correct_so_far, transformed_correct_so_far = 0, 0, 0
        with torch.no_grad():
            for loaded in dataloader:
                original = loaded['original']
                transformed = loaded['transformed']
                labels = loaded['label']
                original_inputs, inputs, labels = original.float().to(device), transformed.float().to(
                    device), labels.float().to(device)
                orig_outputs, transformed_outputs = model(original_inputs), model(inputs)
                orig_pred, transformed_pred = torch.argmax(orig_outputs, dim=1), torch.argmax(transformed_outputs, dim=1)
                total_so_far += len(labels)
                orig_correct_so_far += (orig_pred == labels).sum()
                transformed_correct_so_far += (transformed_pred == labels).sum()
                for i in range(len(orig_outputs)):
                    data.append({
                        "orig_pred": int(orig_pred[i]),
                        "transf_pred": int(transformed_pred[i]),
                        "label": int(labels[i]),
                        "image_path": str(loaded['image_path'][i]),
                        "transformation": loaded["transformation"][i],
                        "transformation_parameter": loaded["transformation_parameter"][i].cpu().numpy(),
                        "vd_score": loaded['vd_score'][i].cpu().numpy(),
                        "model": model_name
                    })
                pbar.update(len(labels))
                pbar.set_postfix({
                    "orig_acc": round(float((orig_correct_so_far / total_so_far).cpu()), 4),
                    "trans_acc": round(float((transformed_correct_so_far / total_so_far).cpu()), 4),
                    # "label": int(labels[i])
                    "Model": model_name
                })

    results_df = pd.DataFrame(data)
    print(tabulate.tabulate(results_df.head(), headers=results_df.columns, tablefmt='pretty'))
    output_path = w.eval_df_save_path(w.get_args().output)
    results_df.to_csv(output_path)
    eval_stats = get_eval_stats(results_df)

    print(tabulate.tabulate([
        ["Original Acc", eval_stats.original_accuracy_str],
        ["Transformed Acc", eval_stats.transformed_accuracy_str]
    ]))
    return eval_stats


def val(model: torch.nn.Module, bootstrap_path: Path, batch_size: int, device: torch.device, classes: List[str],
        optimizer: torch.optim.Optimizer, criterion: torch.nn.Module):
    model.eval()
    loss_sum = 0
    for path in bootstrap_path.iterdir():
        dataset = ImageFolderV2(path, classes=classes,
                                transform=IMAGENET_DEFAULT_TRANSFORMATION)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for inputs, labels, filename in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_sum += float(loss.cpu())
    return loss_sum

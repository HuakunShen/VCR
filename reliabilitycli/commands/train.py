import logging
from typing import List

import pathlib2
import torch
from pathlib2 import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from reliabilitycli.commands.eval import val
from reliabilitycli.src.constants import IMAGENET_DEFAULT_TRANSFORMATION
from reliabilitycli.src.dataset import ImageFolderV2
from reliabilitycli.src.dataset_info import ClassificationDatasetInfo, load_dataset_info
from reliabilitycli.src.utils.model import load_custom_model, get_model
from reliabilitycli.src.workspace import Workspace

logger = logging.getLogger(__name__)


def train(model: torch.nn.Module, bootstrap_path: Path, batch_size: int, device: torch.device, classes: List[str],
          optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, lr_scheduler):
    model.train()
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
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.cpu())
    lr_scheduler.step()
    return loss_sum


def retrain_help(criterion: torch.nn.Module, dataset_info: ClassificationDatasetInfo,
                 optimizer: torch.optim.Optimizer, batch_size: int, epochs: int, model: torch.nn.Module,
                 bootstrap_path: pathlib2.Path, device: torch.device, lr_scheduler, val_interval: int = 1):
    writer = SummaryWriter()
    model = model.to(device)
    training_losses = []
    validation_loss = []
    pbar = tqdm(range(epochs), desc="Retraining")
    for epoch in pbar:
        model.train()
        loss = train(model, bootstrap_path, batch_size, device, dataset_info.classes, optimizer, criterion,
                     lr_scheduler)
        training_losses.append(loss)
        writer.add_scalar('Loss/train', loss, epoch)
        if epoch % val_interval == 0:
            model.eval()
            loss = val(model, bootstrap_path, batch_size, device,
                       dataset_info.classes, optimizer, criterion)
            validation_loss.append(loss)
            writer.add_scalar('Loss/val', loss, epoch)
        pbar.update(1)
    writer.close()


def retrain():
    w = Workspace.instance()
    w.setup_workspace()
    w.config.verify_config()

    dataset_info = load_dataset_info()
    logger.info("retrain")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    batch_size, epochs = 2, 5
    if w.config.config['model_name'] == "custom":
        model = load_custom_model().to(device).eval()
    else:
        model = get_model(w.config.config['model_name'])(pretrained=True).to(device).eval()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.1)
    retrain_help(criterion, dataset_info, optimizer, batch_size=2, epochs=20, model=model,
                 bootstrap_path=w.bootstrap_path, device=device, lr_scheduler=lr_scheduler)

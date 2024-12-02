import torch
import importlib
from typing import Union
from pathlib2 import Path

from reliabilitycli.src.constants import TORCHVISION_MODEL_MAPPING
from reliabilitycli.src.workspace import Workspace

def get_model(model_name: str) -> Union[None, torch.nn.Module]:
    if model_name not in TORCHVISION_MODEL_MAPPING:
        return None
    return TORCHVISION_MODEL_MAPPING[model_name]

def save_model_weights(model: torch.nn.Module, save_path: Union[Path, str]):
    torch.save(model.state_dict(), save_path)

def load_model_weights(model: torch.nn.Module, weight_path: Union[Path, str]) -> torch.nn.Module:
    return model.load_state_dict(torch.load(weight_path))

def save_model(model: torch.nn.Module, save_path: Union[Path, str]):
    torch.save(model, save_path)

def load_model(model_path: Union[Path, str]) -> torch.nn.Module:
    return torch.load(model_path)

def load_custom_model():
    w = Workspace.instance()
    w.get_logger().info("Loading custom model")
    mod = importlib.import_module("workspace.model", str(w.get_workspace_path() / 'model.py'))
    return mod.model

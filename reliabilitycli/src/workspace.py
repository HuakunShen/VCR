from __future__ import annotations

import argparse
import re
import sys

import torch
import yaml
import json
import logging
import datetime
from typing import Dict, List, Tuple
from yaml import CLoader
from pathlib2 import Path
import numpy as np
import random
import importlib.util

from reliabilitycli.src import constants
from reliabilitycli.src.constants import CIFAR10, IMAGENET
from reliabilitycli.src.constants.dataset import IMAGENET16CLASS
from reliabilitycli.src.utils.log import CustomFormatter


class Config:
    """
    Configuration class holding all workspace configurations.
    Will also be responsible for generating and parsing configuration file
    """
    version = 1
    # the configuration template that will be saved during initialization
    config_template = {
        "config_version": version,
        "random_seed": 0,
        "logger": {
            "file_logger_format": constants.FILE_LOGGER_FORMAT,
            "console_logger_format": constants.CONSOLE_LOGGER_FORMAT
        },
        "transformation": "gaussian_noise",
        "dataset_dir": "<TODO: Enter Path to datasets here>",
        "dataset_name": IMAGENET,
        "model": "custom",
        "sample_result_csv_filename": "sample_results.csv",
        "model_filename": "model.py"
    }

    def __init__(self):
        w = Workspace.instance()
        logger = w.get_logger()
        self.dataset_dir = None
        self._config_path = w.get_workspace_path() / 'config.yaml'
        logger.info(f"init config, config path={self._config_path}")
        print(w.get_args())
        if not self._config_path.exists() or (w.get_args().command == 'gen-config' and w.get_args().overwrite):
            # overwrite option is only available in gen-config command
            if w.get_args().overwrite:
                logger.info("overwrite option is on, config file will be overwritten")
            else:
                logger.info("config file doesn't exists, creating a new one")

            self.dump_config()
        logger.info("loading configuration file")
        self.config = self.load_config()
        logger.info("Config File Content=\n" + json.dumps(self.config, indent=2, sort_keys=True))

    def get_config_content(self):
        return self.config

    def load_config(self) -> Dict:
        """
        load configuration file as a python dict
        :return: Configuration Dictionary
        """
        with open(str(self._config_path), 'r') as f:
            config = yaml.load(f, Loader=CLoader)
            if config['config_version'] != Config.version:
                w = Workspace.instance()
                template_file_path = w.get_workspace_path() / 'config-template.yaml'
                msg = f"configuration file version={config['config_version']} isn't the same as " \
                      f"latest version={Config.version}, use the latest version. A template file will be saved to " \
                      f"{str(template_file_path)}"
                w.get_logger().error(msg)
                self.dump_config(template_file_path)
                raise ValueError(msg)
        return config

    def verify_config(self) -> bool:
        """
        Verify if the configuration file is valid
        This method should be run after self.load_config has been called
        :return: True if the configuration file is valid, False otherwise
        """
        if self.config is None:
            return False
        # check dataset path
        w = Workspace.instance()
        msg = None
        if not Path(self.config['dataset_dir']).exists():
            msg = f"Dataset Path={self.config['dataset_dir']} doesn't exist"
        self.dataset_dir = Path(self.config['dataset_dir'])
        if not 'dataset_name' in self.config.keys():
            msg = "Missing dataset_name in config.yml"
        if self.config['dataset_name'] not in [IMAGENET, CIFAR10, IMAGENET16CLASS]:
            msg = f"Invalid dataset_name in config.yml, choose from {IMAGENET} and {CIFAR10}"
        if msg is not None:
            w.get_logger().error(msg)
            raise ValueError(msg)
        # (self.dataset_dir / 'ImageNet').mkdir(parents=True, exist_ok=True)
        return True

    @property
    def dataset_name(self) -> str:
        return self.config['dataset_name']

    @property
    def transformation(self) -> str:
        return self.config['transformation']

    @property
    def model_py_filename(self) -> str:
        if 'model_filename' in self.config:
            return self.config['model_filename']
        else:
            return 'model.py'

    def dump_config(self, path: Path = None) -> None:
        """
        Dump template configuration to a json file
        :param path: where to save the config json file
        :return: None
        """
        with open(str(path or self._config_path), 'w') as f:
            yaml.dump(Config.config_template, f)


class Workspace:
    """
    Have to call initialize() after creating a workspace
    """

    __instance = None

    @classmethod
    def instance(cls) -> Workspace:
        """
        Implementation for singleton design pattern
        :return: a singleton instance of Workspace
        """
        return cls() if cls.__instance is None else cls.__instance

    def __init__(self):
        """
        Initializer for the singleton class. This method is not allowed to be called more than once.
        This initliazer only set instance variables to None, initialize() method should be called.
        """
        if Workspace.__instance is not None:
            raise Exception("This class is a singleton and should not be init twice!")
        else:
            Workspace.__instance = self
            self._project_name = None
            self._logger = None
            self._config = None
            self._args = None
            self._workspace_path = None

    def initialize(self, workspace_path: str, project_name: str, config: Config = None,
                   args: argparse.Namespace = None) -> Workspace:
        """
        Method used to initialize instance variables
        :param workspace_path: workspace directory
        :param project_name: name of project or current experiment
        :param config: Configuration object
        :param args: a copy of cli arguments received
        :return: self, following Chain-of-responsibility pattern
        """
        self._project_name = re.sub('[\s,]', '', project_name)
        self._logger = logging.getLogger(project_name)
        self._args = args
        self._workspace_path = Path(workspace_path).absolute()
        self.setup_logger().setup_workspace()

        if config is None:
            config = Config()
        self._config = config
        self._config.load_config()
        np.random.seed(config.config['random_seed'])
        random.seed(config.config['random_seed'])
        return self

    @property
    def config(self):
        return self._config

    def load_model(self) -> torch.nn.Module:
        model_py_filepath = self._workspace_path / self.config.model_py_filename
        mod = importlib.util.spec_from_file_location('model', str(model_py_filepath)).loader.load_module()
        if not hasattr(mod, 'model'):
            raise ModuleNotFoundError("model not defined in model.py in workspace")
        return mod.model
    
    def load_models(self) -> List[Tuple[str, torch.nn.Module]]:
        model_py_filepath = self._workspace_path / self.config.model_py_filename
        mod = importlib.util.spec_from_file_location('model', str(model_py_filepath)).loader.load_module()
        if not hasattr(mod, 'models'):
            raise ModuleNotFoundError("model not defined in model.py in workspace")
        return mod.models

    def set_config(self, config) -> Workspace:
        """
        :param config: setter for config instance variable
        :return: self, following Chain-of-responsibility pattern
        """
        self._config = config
        return self

    def get_config(self):
        """
        :return: configuration object
        """
        return self._config

    def set_args(self, args) -> Workspace:
        """
        setter for args
        :param args: cli arguments
        :return: self, following Chain-of-responsibility pattern
        """
        self._args = args
        return self

    def get_args(self):
        """
        :return: cli arguments of current run
        """
        return self._args

    def get_logger(self) -> logging.Logger:
        """
        :return: configured logger of this workspace
        """
        if self._logger is None:
            raise ValueError("logger no initialized")
        return self._logger

    def get_workspace_path(self) -> Path:
        """
        :return: path to the workspace of current run
        """
        return self._workspace_path

    @property
    def bootstrap_path(self) -> Path:
        """
        :return: path to bootstrap directory
        """
        return self._workspace_path / 'bootstrap'

    @property
    def sample_df_save_path(self) -> Path:
        if self.get_args().sample_csv_output is None:
            return self._workspace_path / self.config.config['sample_result_csv_filename']
        if self.get_args().sample_csv_output == self.config.config['sample_result_csv_filename']:
            return self._workspace_path / self.config.config['sample_result_csv_filename']
        else:
            return self._workspace_path / self.get_args().sample_csv_output

    def eval_df_save_path(self, filename: str = None) -> Path:
        filename = 'eval_results.csv' if filename is None else filename
        return (self._workspace_path / filename).absolute()
    
    @property
    def report_dir_path(self) -> Path:
        path: Path = self.get_workspace_path() / 'report'
        path = path.absolute()
        # path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def original_path(self) -> Path:
        """
        We will compare model's performance between original dataset and generated datasets.
        The original dataset is not a full dataset, but original images that are used to generate new bootstrapping
        datasets.
        :return: path to the directory in workspace storing sym links to original dataset images
        """
        return self._workspace_path / 'original'

    def create_workspace_dir(self) -> Workspace:
        """
        Create workspace directory and create .workspace.txt to indicate that this is a workspace directory
        :return: self, following Chain-of-responsibility pattern
        """
        workspace_indicator = self._project_name + ".workspace.txt"
        print("self._workspace_path")
        print(self._workspace_path)
        self._workspace_path.mkdir(parents=True, exist_ok=True)
        with open(str(self._workspace_path / workspace_indicator), 'w') as f:
            f.write(f"created at {datetime.datetime.now()}")
        return self

    def setup_workspace(self) -> Workspace:
        """
        Setup workspace directory
        :return: self, following Chain-of-responsibility pattern
        """
        workspace_indicator = self._project_name + ".workspace.txt"
        if not self._workspace_path.exists():
            self._logger.info("workspace doesn't exist, creating the workspace directory")
            self.create_workspace_dir()
        else:
            if not self._workspace_path.is_dir():
                msg = "Workspace path exists and isn't a directory, double check your path"
                self._logger.error(msg)
                raise ValueError(msg)
            # check if the existing directory belongs to this project, if so, we can use it
            if (self._workspace_path / workspace_indicator).exists():
                self._logger.debug("workspace exists, continue")
            else:
                msg = "Workspace path exists but doesn't seem to belong to this project, please remove the directory"
                self._logger.error(msg)
                raise ValueError(msg)
        self._logger.info("setup_workspace finished")
        return self

    def setup_logger(self) -> Workspace:
        """
        Set up logger, add console and file handler
        :return: self, following Chain-of-responsibility pattern
        """
        self._logger.setLevel(logging.DEBUG)
        format_ = self._config.get_config_content()['logger'][
            'console_logger_format'] if self._config and 'console_logger_format' in self._config.get_config_content()[
            'logger'].keys() else constants.CONSOLE_LOGGER_FORMAT
        stdout_formatter = logging.Formatter(format_)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(CustomFormatter())
        format_ = self._config.get_config_content()['logger'][
            'file_logger_format'] if self._config and 'file_logger_format' in self._config.get_config_content()[
            'logger'].keys() else constants.FILE_LOGGER_FORMAT
        file_formatter = logging.Formatter(format_)
        if not self._workspace_path.exists():
            self.create_workspace_dir()
        file_handler = logging.FileHandler(str(self._workspace_path / (self._project_name + '.log')))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        self._logger.addHandler(file_handler)
        self._logger.addHandler(stdout_handler)
        self._logger.info("logger finished setup")
        return self

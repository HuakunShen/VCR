from torchvision import models
from torchvision.models import resnet50
from reliabilitycli.src.dataset_info import Cifar10DatasetInfo
from reliabilitycli.src.imagegen.generator import ImageGenerator
from reliabilitycli.src.workspace import Workspace
from robustbench.utils import load_model

import torch
from robustbench.data import load_imagenet3dcc
from robustbench.utils import clean_accuracy, load_model
from robustbench.eval import benchmark


def experiment():
    # model = resnet50(pretrained=True)
    model = load_model('Standard_R50', dataset='imagenet', threat_model='corruptions')
    print(model)
    # model = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra',
    #                    dataset='cifar10',
    #                    threat_model='Linf')

    # Evaluate the Linf robustness of the model using AutoAttack
    # clean_acc, robust_acc = benchmark(model,
    #                                   dataset='cifar10',
    #                                   threat_model='Linf')
    # print(clean_acc, robust_acc)
    # print(model)


if __name__ == "__main__":
    experiment()

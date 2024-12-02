import torchvision
from torch.utils.data import DataLoader

IMAGENET_NORMALIZE = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])

IMAGENET_DEFAULT_TRANSFORMATION = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    IMAGENET_NORMALIZE,
])
imagenet_data = torchvision.datasets.ImageNet('/home/user/datasets/imagenet', split='val',
                                              transform=IMAGENET_DEFAULT_TRANSFORMATION)

data_loader = DataLoader(imagenet_data, batch_size=4, shuffle=True)
for data in data_loader:
    print(data)

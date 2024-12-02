# Demo

## Coco Bootstrapping

```python
dataset_info = ImagenetDatasetInfo(data_root=datasets_folder / "imagenet/val",
                                   original_data_root=datasets_folder / "imagenet",
                                   original_split='val')
bootstrapper = ImageNetBootstrapper(5, 10, destination=w.get_workspace_path() / 'bootstrap',
                                    dataset_info=dataset_info,
                                    transformation_type=GAUSSIAN_NOISE,
                                    threshold=1.0)
bootstrapper.run().save().validate()
```

## Cifar10 Bootstrapping

```python
dataset_info = Cifar10DatasetInfo(data_root=datasets_folder / 'cifar10' / 'cifar10_pytorch')
bootstrapper = Cifar10Bootstrapper(5, 100, destination=w.get_workspace_path() / 'bootstrap',
                                   dataset_info=dataset_info,
                                   transformation_type=GAUSSIAN_NOISE,
                                   threshold=1.0)
bootstrapper.run().save().validate()
```


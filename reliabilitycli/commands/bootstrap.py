from reliabilitycli.src.bootstrap import ImageNetBootstrapper, Cifar10Bootstrapper
from reliabilitycli.src.constants import IMAGENET, CIFAR10
from reliabilitycli.src.dataset_info import load_dataset_info
from reliabilitycli.src.workspace import Workspace


def bootstrap():
    # python main.py -w ./workspace bootstrap --sample_size 5 --num_sample_iter 2
    w = Workspace.instance()
    w.setup_workspace()
    w.config.verify_config()
    dataset_info = load_dataset_info()

    if w.config.dataset_name == IMAGENET:

        bootstrapper = ImageNetBootstrapper(w.get_args().num_sample_iter, w.get_args().sample_size, workspace=w,
                                            dataset_info=dataset_info,
                                            transformation_type=w.config.transformation,
                                            threshold=1.0)
    elif w.config.dataset_name == CIFAR10:
        (w.config.dataset_dir / 'cifar10' /
         'cifar10_pytorch').mkdir(parents=True, exist_ok=True)
        bootstrapper = Cifar10Bootstrapper(5, 100, workspace=w, dataset_info=dataset_info,
                                           transformation_type=w.config.transformation,
                                           threshold=1.0)
    else:
        raise ValueError(f"{w.config.dataset_name} is not a valid dataset name")
    bootstrapper.run().save().validate()
    bootstrapper.dump()
    print(bootstrapper.bootstrap_df)
    return bootstrapper


def setup_bootstrap_parser(subparsers):
    bootstrap_parser = subparsers.add_parser('bootstrap', help='bootstrap help')
    bootstrap_parser.add_argument(
        '--num_sample_iter', type=int, required=True, help='Number of sample iterations')
    bootstrap_parser.add_argument(
        '--sample_size', type=int, required=True, help='Bootstrap Sample Size')


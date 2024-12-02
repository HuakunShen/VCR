import argparse
import logging

from tabulate import tabulate
from argparse import ArgumentParser

from reliabilitycli.src.dataset_info import Cifar10DatasetInfo
from reliabilitycli.src.dataset_info.factory import dataset_info_factory
from reliabilitycli.src.imagegen.generator import ImageGenerator
from reliabilitycli.src.utils.log import get_custom_logger_formatter
from reliabilitycli.src.workspace import Workspace
from reliabilitycli.src.constants.transformation import TRANSFORMATIONS
from reliabilitycli.src.constants.imagenet_16_class_map import imagenet_16_class_wnid_to_category

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(get_custom_logger_formatter())


def setup_sample_parser(sample_parser: ArgumentParser):
    sample_parser.add_argument(
        '-s', '--size', type=int, required=True, help='Sample Size')
    # sample_parser.add_argument(
    #     '-t', '--transformation', choices=TRANSFORMATIONS, help='Transformation Type')
    sample_parser.add_argument('-o', '--sample-csv-output', type=str,
                               help='Output Filename. (not path, file will be stored in workspace)')


def sample(image_generate_cls: ImageGenerator = ImageGenerator):
    w = Workspace.instance()
    w.setup_workspace()
    w.config.verify_config()
    # print(w.get_args().output)
    # exit(0)
    assert w.get_args().command == "sample", "Wrong Command Handler"
    assert w.get_args().size is not None, "Both size and threshold should be supplied"

    dataset_info = dataset_info_factory(
        w.config.dataset_name, w.config.dataset_dir)

    generator = image_generate_cls(
        w, w.get_args().size, w.config.transformation, dataset_info)
    generator.run().save_df()
    # generator.run().save_df().save()
    logger.info(
        f"Image Generated, metadata saved to {generator.sample_df_save_path}")
    print(tabulate(generator.df.tail(), tablefmt='pretty',
          headers=generator.df.columns))

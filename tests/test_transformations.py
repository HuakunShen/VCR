# PYTHONPATH=$PWD:$PYTHONPATH python tests/test_transformations.py

from reliabilitycli.src.utils.transform import bootstrap_transform
from reliabilitycli.src.constants.transformation import TRANSFORMATIONS, BRIGHTNESS, CONTRAST, DEFOCUS_BLUR, GAUSSIAN_NOISE, GLASS_BLUR
from PIL import Image
from pathlib2 import Path


DIR = Path(__file__).absolute().parent

def test_all_transformations_work():
    """Simply Make sure all transformations can go through"""
    for t in TRANSFORMATIONS:
        bootstrap_transform(Image.open(str(DIR / "sample.JPEG")), GAUSSIAN_NOISE)


if __name__ == "__main__":
    test_all_transformations_work()
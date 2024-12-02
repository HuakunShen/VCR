import numpy as np
from PIL import Image
from typing import Union, Tuple
import random
from PIL.ImageFile import ImageFile
from PIL import ImageOps
from torchvision import transforms

# from ..transformation.imagenet_c_transformations import *
from ..transformation.transformations import glass_blur, defocus_blur, gaussian_blur, Median_Blur, shot_noise, impulse_noise, frost, contrast, Motion_Blur, brightness, jpeg_compression, gaussian_noise
from ..constants.transformation import *
import torchvision.transforms
import albumentations as A



def get_image_based_on_transformation(transformation: str, image_path: str) -> Image:
    """
    The transformation libraries expect different image format, PIL or numpy array.
    This helper function will load image with different function based on the transformation type
    :param transformation: transformation type to use throughout current experiment
    :param image_path: path to image which is to be loaded
    :return: loaded image
    """
    return Image.open(image_path).convert('RGB')  # could be grayscale image with 1 channel, still read as RGB
    # if transformation in [GAUSSIAN_NOISE, FROST, BRIGHTNESS, CONTRAST, JPEG_COMPRESSION, RGB, COLOR_JITTER,
    #                       DEFOCUS_BLUR]:
    #     img = Image.open(image_path).convert('RGB')
    # else:
    #     img = np.asarray(cv2.imread(image_path), dtype=np.float32)
    # return img


def bootstrap_transform(original_image: Image, transformation: str,
                        transformation_level: float = None) -> Tuple[np.ndarray, int]:
    """
    Helper for transforming a given image.
    Given image, transformation type and parameter, return transformed image
    :param original_image: original image
    :param transformation: transformation type
    :param transformation_level: transformation parameter
    :return: transformed image and transformation parameter
    """
    param_index = transformation_level if transformation_level is not None else random.choice(
        range(TRANSFORMATION_LEVEL))
    if transformation == GAUSSIAN_NOISE:
        img2, _ = gaussian_noise(original_image, param_index)
    elif transformation == GAUSSIAN_BLUR:
        img2, _ = gaussian_blur(original_image, param_index)
    elif transformation == SHOT_NOISE:
        img2, _ = shot_noise(original_image, param_index)
    elif transformation == IMPULSE_NOISE:
        img2, _ = impulse_noise(original_image, param_index)
    elif transformation == DEFOCUS_BLUR:
        img2, _ = defocus_blur(original_image, param_index)
    elif transformation == GLASS_BLUR:
        img2, _ = glass_blur(original_image, param_index)
    elif transformation == MOTION_BLUR:
        img2, _ = Motion_Blur(original_image, param_index)
    elif transformation == FROST:
        img2, _ = frost(original_image, param_index)
    elif transformation == CONTRAST:
        img2, _ = contrast(original_image, param_index)
    elif transformation == DEFOCUS_BLUR:
        img2, param = defocus_blur(original_image, param_index)
    elif transformation == FROST:
        img2, _ = frost(original_image, param_index)
    elif transformation == BRIGHTNESS:
        img2, _ = brightness(original_image, param_index)
    elif transformation == GAMMA:
        c = np.linspace(80, 120, TRANSFORMATION_LEVEL)
        img2 = A.gamma_transform(np.array(original_image), gamma=c)
    elif transformation == JPEG_COMPRESSION:
        img2, _ = jpeg_compression(original_image, param_index)
        img2 = np.asarray(img2)
    # elif transformation == COLOR_JITTER:
    #     param_index = random.choice(range(TRANSFORMATION_LEVEL))
    #     img2, _ = Color_jitter(original_image, param_index)
    #     img2 = np.asarray(img2)
    # ============= different transformation types end =============
    else:
        raise ValueError("Invalid Transformation")
    return img2, param_index


# def to_gray(img: Image | np.ndarray):
#     if isinstance(img, Image):
#         return np.array(ImageOps.grayscale(img))
#     else:
#         return rgb2gray(img)


toTensor = torchvision.transforms.ToTensor()
IMAGENET_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
IMAGENET_DEFAULT_TRANSFORMATION = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    torchvision.transforms.ToTensor()
    # IMAGENET_NORMALIZE,
])

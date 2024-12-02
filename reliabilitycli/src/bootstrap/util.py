from reliabilitycli.src.utils.transform import get_image_based_on_transformation, bootstrap_transform
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim


def transform_image(transformation_type: str, image_path: str, threshold: float):
    img = get_image_based_on_transformation(transformation_type, image_path)
    while True:
        img2, param_index = bootstrap_transform(img, transformation_type)
        gray_1, gray_2 = rgb2gray(img), rgb2gray(img2)
        ssim_noise = ssim(gray_1, gray_2,
                          data_range=gray_1.max() - gray_2.min())
        if 1 - ssim_noise < threshold:
            return {
                'transformation': transformation_type,
                'transformation_parameter': param_index,
                'vd_score': 1 - ssim_noise,
                'image': img2
            }

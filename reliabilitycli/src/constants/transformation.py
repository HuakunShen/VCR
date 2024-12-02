"""
this variable contains constant variables related to transformations such as transformation types and default
dataset transformation
"""
import torchvision.transforms as transforms

# noises
GAUSSIAN_NOISE = "gaussian_noise"
SHOT_NOISE = "shot_noise"
IMPULSE_NOISE = "impulse_noise"
# blurs
DEFOCUS_BLUR = "defocus_blur"
GLASS_BLUR = "glass_blur"
MOTION_BLUR = "motion_blur"
GAUSSIAN_BLUR = "gaussian_blur"

# other
FROST = "frost"
CONTRAST = "contrast"
GAMMA = "gamma"

BRIGHTNESS = "brightness"
JPEG_COMPRESSION = "jpeg_compression"
RGB = "RGB"
COLOR_JITTER = "color_jitter"

# TRANSFORMATIONS = [GAUSSIAN_NOISE, FROST, BRIGHTNESS, CONTRAST, JPEG_COMPRESSION, RGB, COLOR_JITTER, DEFOCUS_BLUR]
TRANSFORMATIONS = [GAUSSIAN_NOISE, SHOT_NOISE, IMPULSE_NOISE, DEFOCUS_BLUR, GLASS_BLUR, MOTION_BLUR, FROST, CONTRAST, GAMMA, GAUSSIAN_BLUR]
TRANSFORMATION_LEVEL = 1000

# datasets transformation

IMAGENET_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
IMAGENET_DEFAULT_TRANSFORMATION = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    # IMAGENET_NORMALIZE,
])


TENSOR_TRANSFORMATION = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(224),
    IMAGENET_NORMALIZE,
])


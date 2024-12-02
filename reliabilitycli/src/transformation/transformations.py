from albumentations.augmentations.transforms import *
import albumentations as A
import sys
sys.path.insert(1, '/u/boyue/ICSE2022-resubmission/lib')
# need python 3.7
# to install this, use pip3= install -U albumentations
# to get emboss and sharpen use pip install git+https://github.com/albumentations-team/albumentations.git
import cv2
import numpy as np
import skimage as sk
from skimage.filters import gaussian
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
from io import BytesIO
from PIL import Image
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor
import os, random
from shutil import copy as copy_file
import pickle
import tarfile
import shutil

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB) # not needed for the image generation part
# Constants
TRANSFORMATION_LEVEL = 1000

# /////////////// Distortions ///////////////

def save_array(dest, arr):
	img = Image.fromarray(arr.astype(np.uint8))
	img.save(dest)

def gaussian_noise(x, i):
	#c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
	#cifar c = np.linspace(0.0001, 0.9, TRANSFORMATION_LEVEL);
	c = np.linspace(0.0001, 1, TRANSFORMATION_LEVEL)
	x = np.array(x) / 255.
	return np.clip(x + np.random.normal(size=x.shape, scale=c[i]), 0, 1) * 255, c[i]


#def shot_noise(x, severity=1):
def shot_noise(x, i):
#	c = [60, 25, 12, 5, 3][severity - 1]
	c = np.linspace(1, 5, TRANSFORMATION_LEVEL);
	
	x = np.array(x) / 255.
	return np.clip(np.random.poisson(x * c[i]) / c[i], 0, 1) * 255, c[i]


#def impulse_noise(x, severity=1):
def impulse_noise(x, i):
#	c = [.03, .06, .09, 0.17, 0.27][severity - 1]
	c = np.linspace(0.0, 0.5, TRANSFORMATION_LEVEL);
	
	x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c[i])
	return np.clip(x, 0, 1) * 255, c[i]


#def speckle_noise(x, severity=1):
def speckle_noise(x, i):
#	c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]
	c = [0.15, 0.2, 0.35, 0.45, 0.6]

	x = np.array(x) / 255.
	return np.clip(x + x * np.random.normal(size=x.shape, scale=c[i]), 0, 1) * 255, c[i]

# not inlcuded
def fgsm(x, source_net, severity=1):
	c = [8, 16, 32, 64, 128][severity - 1]

	x = V(x, requires_grad=True)
	logits = source_net(x)
	source_net.zero_grad()
	loss = F.cross_entropy(logits, V(logits.data.max(1)[1].squeeze_()), size_average=False)
	loss.backward()

	return standardize(torch.clamp(unstandardize(x.data) + c / 255. * unstandardize(torch.sign(x.grad.data)), 0, 1))

#def gaussian_blur(x, severity=1):
def gaussian_blur(x, i):
#	c = [1, 2, 3, 4, 6][severity - 1]
	c = np.linspace(0.00001,5,TRANSFORMATION_LEVEL)
	# print(c)
	x = gaussian(np.array(x) / 255., sigma=c[i], channel_axis=2)
	return np.clip(x, 0, 1) * 255, c[i]

#def glass_blur(x, severity=1):
# def glass_blur(x, i):
# 	# sigma, max_delta, iterations
# #	c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
	
# 	sigma = np.linspace(0, 2, TRANSFORMATION_LEVEL)
# 	np.random.shuffle(sigma)
# 	max_delta = np.random.randint(1,10,TRANSFORMATION_LEVEL)
# 	np.random.shuffle(max_delta)
# 	iterations = 1
# 	c = np.stack([sigma, max_delta], 1)
	
# 	x = np.uint8(gaussian(np.array(x) / 255., sigma=c[i], channel_axis=2) * 255)

# 	# locally shuffle pixels
# 	for i in range(iterations):
# 		for h in range(32 - max_delta[i], max_delta[i], -1):
# 			for w in range(32 - max_delta[i], max_delta[i], -1):
# 				dx, dy = np.random.randint(-max_delta[i], max_delta[i], size=(2,))
# 				h_prime, w_prime = h + dy, w + dx
# 				# swap
# 				x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

# 	return np.clip(gaussian(x / 255., sigma=sigma[i], channel_axis=2), 0, 1) * 255, c[i]

def glass_blur(x, i):
	# sigma = np.linspace(0, 8, TRANSFORMATION_LEVEL)
	# np.random.shuffle(sigma)
	# max_delta = np.random.randint(1,15,TRANSFORMATION_LEVEL)
	# np.random.shuffle(max_delta)
	# iterations = 1#random.choice([1, 2])
	
	sigma = np.linspace(0, 5, TRANSFORMATION_LEVEL)
	np.random.shuffle(sigma)
	max_delta = np.random.randint(1,10,TRANSFORMATION_LEVEL)
	np.random.shuffle(max_delta)
	iterations = 1#random.choice([1, 2])
	c = np.stack([sigma, max_delta], 1)

	x = np.uint8(gaussian(np.array(x) / 255., sigma=sigma[i], channel_axis=2) * 255)
	max_delta = [max_delta[i]]

	# locally shuffle pixels
	for j in range(iterations):
		for h in range(32 - max_delta[j], max_delta[j], -1):
			for w in range(32 - max_delta[j], max_delta[j], -1):
				dx, dy = np.random.randint(-max_delta[j], max_delta[j], size=(2,))
				h_prime, w_prime = h + dy, w + dx
				# swap
				x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

	return np.clip(gaussian(x / 255., sigma=sigma[i], channel_axis=2), 0, 1) * 255, c[i]


def disk(radius, alias_blur=0.1, dtype=np.float32):
	if radius <= 8:
		L = np.arange(-8, 8 + 1)
		ksize = (3, 3)
	else:
		L = np.arange(-radius, radius + 1)
		ksize = (5, 5)
	X, Y = np.meshgrid(L, L)
	aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
	aliased_disk /= np.sum(aliased_disk)

	# supersample disk to antialias
	return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

#def defocus_blur(x, severity=1):
def defocus_blur(x, i):
#	c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
	#cifar radius = np.linspace(0, 8, TRANSFORMATION_LEVEL)
	radius = np.linspace(0, 10, TRANSFORMATION_LEVEL)
	np.random.shuffle(radius)
	alias_blur = np.linspace(0, 1, TRANSFORMATION_LEVEL)
	np.random.shuffle(alias_blur)
	c = np.stack([radius, alias_blur], 1)

	x = np.array(x) / 255.
	kernel = disk(radius=c[i][0], alias_blur=c[i][1])

	channels = []
	for d in range(3):
		channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
	channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

	return np.clip(channels, 0, 1) * 255, c[i]

class MotionImage(WandImage):
	def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
		wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)

#def motion_blur(x, severity=1):
def motion_blur_v2(x, i):
#	c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
	
	radius = np.linspace(10, 40, TRANSFORMATION_LEVEL)
	np.random.shuffle(radius)
	sigma = np.linspace(10, 30, TRANSFORMATION_LEVEL)
	np.random.shuffle(sigma)
	c = np.stack([radius, sigma], 1)
	
	output = BytesIO()
	x.save(output, format='PNG')
	x = MotionImage(blob=output.getvalue())

	x.motion_blur(radius=c[i][0], sigma=c[i][1], angle=np.random.uniform(-45, 45))

	x = cv2.imdecode(np.frombuffer(x.make_blob(), np.uint8),
					 cv2.IMREAD_UNCHANGED)

	if x.shape != (32, 32):
		return np.clip(x[..., [2, 1, 0]], 0, 255), c[i]  # BGR to RGB
	else:  # greyscale to RGB
		return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255), c[i]

def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / float(zoom_factor)))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


def zoom_blur(x, i):
	#factor_one = np.linspace(1, 1.5, TRANSFORMATION_LEVEL)
	#np.random.shuffle(factor_one)
	#factor_two = np.linspace(0.00001, 0.05, TRANSFORMATION_LEVEL)
	#np.random.shuffle(factor_two)
	#c = np.arange(1, 1.31, 0.03)
	#c = np.arange(1, factor_one[i], factor_two[i])
	severity = i % 5
	c = [np.arange(1, 1.11, 0.01),
	     np.arange(1, 1.16, 0.01),
	     np.arange(1, 1.21, 0.02),
	     np.arange(1, 1.26, 0.02),
	     np.arange(1, 1.31, 0.03)][severity]

	x = (np.array(x) / 255.).astype(np.float32)
	out = np.zeros_like(x)
	for zoom_factor in c:
		out += clipped_zoom(x, zoom_factor)

	x = (x + out) / (len(c) + 1)
	return np.clip(x, 0, 1) * 255, i


def plasma_fractal(mapsize=256, wibbledecay=3):
	"""
	Generate a heightmap using diamond-square algorithm.
	Return square 2d array, side length 'mapsize', of floats in range 0-255.
	'mapsize' must be a power of two.
	"""
	assert (mapsize & (mapsize - 1) == 0)
	maparray = np.empty((mapsize, mapsize), dtype=np.float_)
	maparray[0, 0] = 0
	stepsize = mapsize
	wibble = 100

	def wibbledmean(array):
		return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

	def fillsquares():
		"""For each square of points stepsize apart,
		   calculate middle value as mean of points + wibble"""
		cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
		squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
		squareaccum += np.roll(squareaccum, shift=-1, axis=1)
		maparray[stepsize // 2:mapsize:stepsize,
		stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

	def filldiamonds():
		"""For each diamond of points stepsize apart,
		   calculate middle value as mean of points + wibble"""
		mapsize = maparray.shape[0]
		drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
		ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
		ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
		lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
		ltsum = ldrsum + lulsum
		maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
		tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
		tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
		ttsum = tdrsum + tulsum
		maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

	while stepsize >= 2:
		fillsquares()
		filldiamonds()
		stepsize //= 2
		wibble /= wibbledecay

	maparray -= maparray.min()
	return maparray / maparray.max()

#def fog(x, severity=1):
def fog(x, i):
# c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]
	
	scale = np.linspace(1.5, 9, TRANSFORMATION_LEVEL)
	np.random.shuffle(scale)
	wibbledecay = np.linspace(1.5, 5, TRANSFORMATION_LEVEL)
	np.random.shuffle(wibbledecay)
	c = np.stack([scale, wibbledecay], 1)

	x = np.array(x) / 255.
	max_val = x.max()
	#print(x.shape)
	h,w,ch = x.shape
	plasma_size = w.bit_length()
	x += c[i][0] * plasma_fractal(mapsize=2**plasma_size, wibbledecay=c[i][1])[:h, :w][..., np.newaxis]
	return np.clip(x * max_val / (max_val + c[i][0]), 0, 1) * 255, c[i]

#def frost(x, severity=1):
def frost(x, i):
	#c = [(1, 0.4),
	#	 (0.8, 0.6),
	#	 (0.7, 0.7),
	#	 (0.65, 0.7),
	#	 (0.6, 0.75)]
	
	scale = np.linspace(0.01, 1, TRANSFORMATION_LEVEL)
	np.random.shuffle(scale)
	constant = np.linspace(0.01, 1, TRANSFORMATION_LEVEL)
	np.random.shuffle(constant)
	c = np.stack([scale, constant], 1)  
	idx = np.random.randint(5)
	filename = ['frost1.png', 'frost2.png', 'frost3.png', 'frost4.jpeg', 'frost5.jpeg', 'frost6.jpeg'][idx]
	frost = Image.open(filename)

	#print(frost)
	x = np.asarray(x)
	h,w,ch = x.shape
	frost = np.asarray(frost.resize((w, h)))
	# randomly crop and convert to rgb
	frost = frost[..., [2, 1, 0]]
	#x_start, y_start = np.random.randint(0, frost.shape[0] - 224), np.random.randint(0, frost.shape[1] - 224)
	#frost = frost[x_start:x_start + 224, y_start:y_start + 224][..., [2, 1, 0]]

	return np.clip(c[i][0] * x + c[i][1] * frost, 0, 255), c[i]

def snow(x, i):
	#c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
	#	 (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
	#	 (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
	#	 (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
	#	 (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]
	
	location = np.linspace(0, 0.6, TRANSFORMATION_LEVEL)
	np.random.shuffle(location)
	scale = np.linspace(0, 0.5, TRANSFORMATION_LEVEL)
	np.random.shuffle(scale)
	zoom = np.linspace(0, 5, TRANSFORMATION_LEVEL)
	np.random.shuffle(zoom)
	snow_layer_threshold = np.linspace(0.5, 1, TRANSFORMATION_LEVEL)
	np.random.shuffle(snow_layer_threshold)
	blur_radius = np.linspace(10, 15, TRANSFORMATION_LEVEL)
	np.random.shuffle(blur_radius)
	blur_sigma = np.linspace(5, 15, TRANSFORMATION_LEVEL)
	np.random.shuffle(blur_sigma)
	layer_scale = np.linspace(0.5, 1, TRANSFORMATION_LEVEL)
	np.random.shuffle(layer_scale)
	c = np.stack([location, scale, zoom, snow_layer_threshold, blur_radius, blur_sigma, layer_scale], 1)
	
	x = np.array(x, dtype=np.float32) / 255.
	h,w,ch = x.shape
	snow_layer = np.random.normal(size=(w,w), loc=c[i][0], scale=c[i][1])  # [:2] for monochrome
	snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[i][2])
	
	snow_layer[snow_layer < c[i][3]] = 0
	
	snow_layer = Image.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
	output = BytesIO()
	snow_layer.save(output, format='PNG')
	snow_layer = MotionImage(blob=output.getvalue())

	snow_layer.motion_blur(radius=c[i][4], sigma=c[i][5], angle=np.random.uniform(-135, -45))

	snow_layer = cv2.imdecode(np.frombuffer(snow_layer.make_blob(), np.uint8),
							  cv2.IMREAD_UNCHANGED) / 255.
	snow_layer = snow_layer[..., np.newaxis]	
	x = c[i][6] * x + (1 - c[i][6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(h, w, 1) * 1.5 + 0.5)
	snow_layer = snow_layer[w//2-h//2:w//2+(h-h//2), 0:w]
	return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255, i

# not used
def spatter(x, severity=1):
	c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
		 (0.65, 0.3, 3, 0.68, 0.6, 0),
		 (0.65, 0.3, 2, 0.68, 0.5, 0),
		 (0.65, 0.3, 1, 0.65, 1.5, 1),
		 (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
	x = np.array(x, dtype=np.float32) / 255.

	liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

	liquid_layer = gaussian(liquid_layer, sigma=c[2])
	liquid_layer[liquid_layer < c[3]] = 0
	if c[5] == 0:
		liquid_layer = (liquid_layer * 255).astype(np.uint8)
		dist = 255 - cv2.Canny(liquid_layer, 50, 150)
		dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
		_, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
		dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
		dist = cv2.equalizeHist(dist)
		#     ker = np.array([[-1,-2,-3],[-2,0,0],[-3,0,1]], dtype=np.float32)
		#     ker -= np.mean(ker)
		ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
		dist = cv2.filter2D(dist, cv2.CV_8U, ker)
		dist = cv2.blur(dist, (3, 3)).astype(np.float32)

		m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
		m /= np.max(m, axis=(0, 1))
		m *= c[4]

		# water is pale turqouise
		color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
								238 / 255. * np.ones_like(m[..., :1]),
								238 / 255. * np.ones_like(m[..., :1])), axis=2)

		color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
		x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

		return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
	else:
		m = np.where(liquid_layer > c[3], 1, 0)
		m = gaussian(m.astype(np.float32), sigma=c[4])
		m[m < 0.8] = 0
		#         m = np.abs(m) ** (1/c[4])

		# mud brown
		color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
								42 / 255. * np.ones_like(x[..., :1]),
								20 / 255. * np.ones_like(x[..., :1])), axis=2)

		color *= m[..., np.newaxis]
		x *= (1 - m[..., np.newaxis])

		return np.clip(x + color, 0, 1) * 255

#def contrast(x, c):
#	c = c/TRANSFORMATION_LEVEL
#	x = np.array(x) / 255.
#	means = np.mean(x, axis=(0, 1), keepdims=True)
#	return np.clip((x - means) * c + means, 0, 1) * 255, c

def contrast(x, i):
#def contrast(x, severity=1):
	#c = [0.4, .3, .2, .1, .05]
	# cifar-10 c = np.linspace(0.01, 0.9, TRANSFORMATION_LEVEL)
	c = np.linspace(0.0001, 0.3, TRANSFORMATION_LEVEL)
	x = np.array(x) / 255.
	means = np.mean(x, axis=(0, 1), keepdims=True)
	return np.clip((x - means) * c[i] + means, 0, 1) * 255, c[i]


#def brightness(x, severity=1):
def brightness(x, i):
	#c = [.1, .2, .3, .4, .5]
	c = np.linspace(0, 1, TRANSFORMATION_LEVEL)	 

	x = np.array(x) / 255.
	x = sk.color.rgb2hsv(x)
	x[:, :, 2] = np.clip(x[:, :, 2] + c[i], 0, 1)
	x = sk.color.hsv2rgb(x)

	return np.clip(x, 0, 1) * 255, c[i]

#def saturate(x, severity=1):
def saturate(x, i):
	c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)]

	x = np.array(x) / 255.
	x = sk.color.rgb2hsv(x)
	x[:, :, 1] = np.clip(x[:, :, 1] * c[i][0] + c[i][1], 0, 1)
	x = sk.color.hsv2rgb(x)

	return np.clip(x, 0, 1) * 255, c[i]


#def jpeg_compression(x, severity=1):
def jpeg_compression(x, i):
	#c = [25, 18, 15, 10, 7][severity - 1]
	#c = [25, 18, 15, 10, 7]
	#c = list(range(1,(TRANSFORMATION_LEVEL+1)))
	#c = list(range(1,(10+1))) * int(TRANSFORMATION_LEVEL/10)
	c = np.random.randint(1, 50, TRANSFORMATION_LEVEL)
	output = BytesIO()
	x.save(output, 'JPEG', quality=int(c[i]))
	x = Image.open(output)

	return x, c[i]


#def pixelate(x, severity=1):
def pixelate(x, i):
	c = np.linspace(0.00001, 1, TRANSFORMATION_LEVEL)	 
	#c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

	x = x.resize((int(224 * c[i]), int(224 * c[i])), Image.BOX)
	x = x.resize((224, 224), Image.BOX)

	return x, i


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
# this is geometric, let's not use it
def elastic_transform(image, severity=1):
	c = [(244 * 2, 244 * 0.7, 244 * 0.1),   # 244 should have been 224, but ultimately nothing is incorrect
		 (244 * 2, 244 * 0.08, 244 * 0.2),
		 (244 * 0.05, 244 * 0.01, 244 * 0.02),
		 (244 * 0.07, 244 * 0.01, 244 * 0.02),
		 (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]

	image = np.array(image, dtype=np.float32) / 255.
	shape = image.shape
	shape_size = shape[:2]

	# random affine
	center_square = np.float32(shape_size) // 2
	square_size = min(shape_size) // 3
	pts1 = np.float32([center_square + square_size,
					   [center_square[0] + square_size, center_square[1] - square_size],
					   center_square - square_size])
	pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
	M = cv2.getAffineTransform(pts1, pts2)
	image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

	dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
				   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
	dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
				   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
	dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

	x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
	indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
	return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255



# /////////////// End Distortions ///////////////


# /////////////// Begin Albumentation ///////////////
# note that they use different image libraries
# below are the transformations and an example of how to run them
# for the experiment, I think it's eaiser to create a pickle file to save a dict of image names to replays because some of the parameters are really long

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ColorJitter
def Color_jitter (x, i):
	#b = np.linspace(0, 1, 100)
	color_jitter = A.ReplayCompose([ColorJitter(brightness=1, contrast=1, saturation=1, hue=1, always_apply=True)])
	transformed_img = color_jitter(image=x)
	# to get the parameters that are actually used
	arguments = str(transformed_img['replay']['transforms'][0]['params']['contrast']) + ", " + str(transformed_img['replay']['transforms'][0]['params']['saturation']) + ", " + str(transformed_img['replay']['transforms'][0]['params']['hue'])
	
	return transformed_img['image'], arguments
	
#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RGBShift
def RGB (x,i):
	RGB_Shift = A.ReplayCompose([RGBShift(r_shift_limit=200, g_shift_limit=200, b_shift_limit=200, always_apply=True)])
	transformed_img = RGB_Shift(image=x)
	# to get the parameters that are actually used
	arguments = str(transformed_img['replay']['transforms'][0]['params']['r_shift']) + ", " +  str(transformed_img['replay']['transforms'][0]['params']['g_shift']) +", " +  str(transformed_img['replay']['transforms'][0]['params']['b_shift'])
	
	return transformed_img['image'], arguments
	
	
#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomBrightnessContrast
def Random_Brightness_Contrast (x, i):
	#c = [(0.1, 0.1), (0.15, 0.15), (0.2, 0.2), (0.25, 0.25), (0.3, 0.3)]
	RBC = A.ReplayCompose([RandomBrightnessContrast(brightness_limit=1, contrast_limit=1, brightness_by_max=True, always_apply=True)])
	transformed_img = RBC(image=x)
	# to get the parameters that are actually used
	arguments = str(transformed_img['replay']['transforms'][0]['params']['alpha']) + ", " + str(transformed_img['replay']['transforms'][0]['params']['beta'])
	
	return transformed_img['image'], arguments
	
#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomGamma
def Random_Gamma (x, i):
	#c = [(80, 120), (90, 130), (100, 140), (120, 150), (130, 160)]
	RG = A.ReplayCompose([RandomGamma(gamma_limit=(0, 300), always_apply=True)])
	transformed_img = RG (image=x)
	# to get the parameters that are actually used
	arguments = str(transformed_img['replay']['transforms'][0]['params']['gamma'])
	
	return transformed_img['image'], arguments
	
#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomSunFlare
def Random_Sun_Flare (x, i):
	RSF = A.ReplayCompose([RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=6, num_flare_circles_upper=10, src_radius=150, src_color=(255, 255, 255), always_apply=True)])
	transformed_img = RSF(image=x)
	# to get the parameters that are actually used
	arguments = len(transformed_img['replay']['transforms'][0]['params']['circles'])
	
	return transformed_img['image'], str(arguments)
	
To_Gray = ToGray(p=1)

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GlassBlur
def Glass_blur_two (x, i):
	c = [(0.7, 4, 2), (0.7, 8, 3), (0.7, 9, 4), (0.7, 10, 5), (0.7, 11, 6)]
	glass_blur = A.ReplayCompose([GlassBlur(sigma=0.7, max_delta=4, iterations=2, always_apply=True, mode='fast')])
	transformed_img = glass_blur(image=x)
	# to get the parameters that are actually used
	arguments = str(transformed_img['replay']['transforms'][0]['params']['ksize']) + ", " + str(transformed_img['replay']['transforms'][0]['params']['dxy'])
	
	return transformed_img['image'], arguments
	
#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomFog
def Random_Fog (x, i):
	#c = [(0.3, 1, 0.08), (0.4, 1, 0.085), (0.45, 1, 0.09), (0.5, 1, 0.1), (0.55, 1, 0.15)]
	random_fog = A.ReplayCompose([RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, always_apply=True)])
	transformed_img = random_fog(image=x)
	# to get the parameters that are actually used
	arguments = str(transformed_img['replay']['transforms'][0]['params']['fog_coef'])
	
	return transformed_img['image'], arguments

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomRain
def Random_Rain (x, i):
	#c = [(10, 10, 15, 1), (10, 10, 15, 1), (10, 10, 15, 1), (10, 10, 15, 1), (10, 10, 15, 1)]
	rr = A.ReplayCompose([RandomRain(slant_lower=10, slant_upper=10, drop_length=15, drop_width=1, drop_color=(200, 200, 200), blur_value=5, brightness_coefficient=0.5, rain_type=None, always_apply=True)])
	transformed_img = rr(image=x)
	# to get the parameters that are actually used
	arguments = str(transformed_img['replay']['transforms'][0]['params']['drop_length'])
	
	return transformed_img['image'], arguments

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomSnow
def Random_Snow (x, i):
	random_snow = A.ReplayCompose([RandomSnow(snow_point_lower=0.1, snow_point_upper=0.2, brightness_coeff=3, always_apply=True)])
	transformed_img = random_snow(image=x)
	# to get the parameters that are actually used
	arguments = str(transformed_img['replay']['transforms'][0]['params']['drop_length']) + ", " + str(transformed_img['replay']['transforms'][0]['params']['snow_drops'])
	
	return transformed_img['image'], arguments
	

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Blur
def Blur_wrap (x, i):
	blur = A.ReplayCompose([Blur (blur_limit=15, always_apply=True)])
	transformed_img = blur(image=x)
	# to get the parameters that are actually used
	arguments = str(transformed_img['replay']['transforms'][0]['params']['ksize'])
	
	return transformed_img['image'], arguments


#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Downscale
def Down_scale (x, i):
	c = np.linspace(0.01, 0.99, 100)
	downscale = A.ReplayCompose([Downscale(scale_min=c[i], scale_max=c[i], interpolation=0, always_apply=True)])
	transformed_img = downscale(image=x)
	# to get the parameters that are actually used
	arguments = str(transformed_img['replay']['transforms'][0]['params']['scale']) + ", " +  str(transformed_img['replay']['transforms'][0]['params']['interpolation'])
	
	return transformed_img['image'], arguments

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GaussNoise
def Gaussian_noise_v2 (x, i): 
	gaussian_noise = A.ReplayCompose([GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=True)])
	transformed_img = gaussian_noise(image=x)
	# to get the parameters that are actually used
	arguments = str(transformed_img['replay']['transforms'][0]['params']['gauss'])
	
	return transformed_img['image'], arguments

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GaussianBlur
def Gaussian_blur (x, i):
	gaussian_blur = A.ReplayCompose([GaussianBlur(blur_limit=(3, 19), sigma_limit=(0,5), always_apply=True)])
	transformed_img = gaussian_blur(image=x)
	# to get the parameters that are actually used
	arguments = str(transformed_img['replay']['transforms'][0]['params']['ksize']) + ", " +  str(transformed_img['replay']['transforms'][0]['params']['sigma'])
	
	return transformed_img['image'], arguments

# TODO remove	
#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GaussianBlur
def Gaussian_blur_v1 (x, i):
	gaussian_blur = A.ReplayCompose([GaussianBlur(blur_limit=(3, 19), sigma_limit=(0,1.4), always_apply=True)])
	transformed_img = gaussian_blur(image=x)
	# to get the parameters that are actually used
	arguments = str(transformed_img['replay']['transforms'][0]['params']['ksize']) + ", " +  str(transformed_img['replay']['transforms'][0]['params']['sigma'])
	
	return transformed_img['image'], arguments
	

# TODO remove	
#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GaussianBlur
def Gaussian_blur_v2 (x, i):
	gaussian_blur = A.ReplayCompose([GaussianBlur(blur_limit=(3, 19), sigma_limit=(1.4,5), always_apply=True)])
	transformed_img = gaussian_blur(image=x)
	# to get the parameters that are actually used
	arguments = str(transformed_img['replay']['transforms'][0]['params']['ksize']) + ", " +  str(transformed_img['replay']['transforms'][0]['params']['sigma'])
	
	return transformed_img['image'], arguments

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ImageCompression
def Image_Compression (x, i):
	image_compression = A.ReplayCompose([ImageCompression (quality_lower=5, quality_upper=100, compression_type=1, always_apply=True)])
	transformed_img = image_compression(image=x)
	# to get the parameters that are actually used
	arguments = str(transformed_img['replay']['transforms'][0]['params']['quality']) + ", " + str(transformed_img['replay']['transforms'][0]['params']['image_type'])
	
	return transformed_img['image'], arguments

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.MedianBlur
def Median_Blur (x, i):
	median_blur = A.ReplayCompose([MedianBlur(blur_limit=21, always_apply=True)])
	transformed_img = median_blur(image=x)
	# to get the parameters that are actually used
	arguments = str(transformed_img['replay']['transforms'][0]['params']['ksize'])
	
	return transformed_img['image'], arguments

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.MotionBlur
def Motion_Blur (x, i):
	motion_blur = A.ReplayCompose([MotionBlur(blur_limit=50,always_apply=True)])
	transformed_img = motion_blur(image=x)
	# to get the parameters that are actually used
	arguments = transformed_img['replay']['transforms'][0]['params']['kernel'].max()
	
	return transformed_img['image'], str(arguments)
	
#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.HueSaturationValue
def HUE_saturation (x, i):
	hue_saturation = A.ReplayCompose([HueSaturationValue(hue_shift_limit=(100), sat_shift_limit=100, val_shift_limit=100, always_apply=True)])
	transformed_img = hue_saturation(image=x)
	# to get the parameters that are actually used
	arguments = str(transformed_img['replay']['transforms'][0]['params']['hue_shift']) + ", " + str(transformed_img['replay']['transforms'][0]['params']['sat_shift']) + ", " + str(transformed_img['replay']['transforms'][0]['params']['val_shift'])
	
	return transformed_img['image'], arguments
	
#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ISONoise
def ISO_noise (x, i):
	ISO_noise = A.ReplayCompose([ISONoise(color_shift=(0.01, 1), intensity=(0.5, 5), always_apply=True)])
	transformed_img = ISO_noise(image=x)
	# to get the parameters that are actually used
	#print(transformed_img['replay']['transforms'][0]['params'])
	arguments = str(transformed_img['replay']['transforms'][0]['params']['color_shift']) + ", " + str(transformed_img['replay']['transforms'][0]['params']['intensity']) + ", " +  str(transformed_img['replay']['transforms'][0]['params']['random_state'])
	
	return transformed_img['image'], arguments
	
#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomToneCurve
def Random_Tone_Curve (x, i):
	tone_curve = A.ReplayCompose([RandomToneCurve(scale=(0.1), always_apply=True)])
	transformed_img = tone_curve(image=x)
	# to get the parameters that are actually used
	arguments = str(transformed_img['replay']['transforms'][0]['params']['low_y']) + ", " + str(transformed_img['replay']['transforms'][0]['params']['high_y'])
	
	return transformed_img['image'], arguments

def apply_uniform_noise(image, i, rng=None):
	"""
	Taken from https://github.com/rgeirhos/generalisation-humans-DNNs/blob/master/code/image_manipulation.py
	"""    
	high_and_low = np.linspace(0.01, 1, TRANSFORMATION_LEVEL)
	choices = random.sample(high_and_low, 2)
	high = max(choices)
	low = min(choices)


	nrow = image.shape[0]
	ncol = image.shape[1]
	nch = image.shape[2]   
	
	image = image / 255 
	
	image = image + get_uniform_noise(low, high, nrow, ncol, nch, rng)    #clip values
	
	image = np.where(image < 0, 0, image)
	image = np.where(image > 1, 1, image)    
	
	assert is_in_bounds(image, 0, 1), "values <0 or >1 occurred"   
	
	image = image * 255
	return image
	
def get_uniform_noise(low, high, nrow, ncol, nch, rng=None):
	"""
	Taken from https://github.com/rgeirhos/generalisation-humans-DNNs/blob/master/code/image_manipulation.py
	"""    
	if rng is None:
		return np.random.uniform(low=low, high=high,
								 size=(nrow, ncol, nch))
	else:
		return rng.uniform(low=low, high=high,
						   size=(nrow, ncol, nch))

'''

# level of transformation considered
nb_level= range(TRANSFORMATION_LEVEL)
#TODO: randomsnow
# list of albumentations images selected
# full experiment (uncoment)
#transformations_l2 = [Color_jitter, RGB, Random_Brightness_Contrast, Random_Gamma, Random_Sun_Flare, Random_Fog, Random_Rain, Down_scale, Gaussian_blur, Motion_Blur, Image_Compression, Random_Tone_Curve, HUE_saturation, ISO_noise]
# pilote experiment

# TODO remove
nb_level_small = range(100)
for i in nb_level_small:
	files=os.listdir(path)
	orig_image_path = random.choice(files)
	orig_img = cv2.imread('./IMGS/' + orig_image_path)
	# copy original image
	#cv2.imwrite(f"IMG_ORIGINAL/"+orig_image_path, orig_img)
	copy_file ('./IMGS/' + orig_image_path, "IMG_ORIGINAL/"+orig_image_path)
	transformed_img, arguments = Gaussian_blur_v1 (orig_img, i)
	name = orig_image_path + "." + Gaussian_blur.__name__ + "_" + str(i) + '.png'
	# writing the image (commented out)
	cv2.imwrite(f"IMG_TRANSFORMED/"+name, transformed_img['image'])
	#imgs_info += "\n" + name + ", " + orig_image_path + "," + Gaussian_blur.__name__ + ", " + arguments
	imgs_info += "\n" + name + ", " + orig_image_path + "," + Gaussian_blur.__name__ + " , " + arguments

transformations_l2 = [Gaussian_blur_v2]

for test_transformations in transformations_l2:
	for i in nb_level:
		level = i + 100
		files=os.listdir(path)
		orig_image_path = random.choice(files)
		orig_img = cv2.imread('./IMGS/' + orig_image_path)
		# copy original image
		#cv2.imwrite(f"IMG_ORIGINAL/"+orig_image_path, orig_img)
		copy_file ('./IMGS/' + orig_image_path, "IMG_ORIGINAL/"+orig_image_path)
		transformed_img, arguments = test_transformations (orig_img, i)
		name = orig_image_path + "." + test_transformations.__name__ + "_" + str(level) + '.png'
		# writing the image (commented out)
		cv2.imwrite(f"IMG_TRANSFORMED/"+name, transformed_img['image'])
		#imgs_info += "\n" + name + ", " + orig_image_path + "," + test_transformations.__name__ + ", " + arguments
		imgs_info += "\n" + name + ", " + orig_image_path + "," + test_transformations.__name__ + " , " + arguments

f_infos = open("images_transformations_info.csv", "w")
f_infos.write (imgs_info);
f_infos.close()

# number of class per image cropped

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ToFloat
to_float = A.ReplayCompose([ToFloat(max_value=None, always_apply=True)]) # no parameter
# using replay
#image2 = cv2.imread('images/image_3.jpg')
#image2_data = A.ReplayCompose.replay(transformed_img['replay'], image=image2)
# /////////////// End Albumentation ///////////////
'''




from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps, ImageEnhance
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings

from . import Function as F

__all__ = ["Compose", "ToTensor", "ToPILImage", "Normalize", "Resize", "Scale", "CenterCrop", "Pad",
           "Lambda", "RandomApply", "RandomChoice", "RandomOrder", "RandomCrop", "RandomHorizontalFlip",
           "RandomVerticalFlip", "RandomResizedCrop", "RandomSizedCrop", "FiveCrop", "TenCrop", "LinearTransformation",
           "ColorJitter", "RandomRotation", 'Rotation','VerticalFlip', "HorizontalFlip"]

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


# my funtions
class Rotation(object):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        angle = degrees
        return angle

    def __call__(self, img):
        angle = self.get_params(self.degrees)
        return F.rotate(img, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class VerticalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):

        if 0.5 < self.p:
            return F.vflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class HorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if 0.5< self.p:
            return F.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class Affine(object):

        def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
            if isinstance(degrees, numbers.Number):
                self.degrees = degrees

            if translate is not None:
                assert isinstance(translate, (tuple, list)) and len(translate) == 2
                for t in translate:
                    if not (0.0 <= t <= 1.0):
                        raise ValueError("translation values should be between 0 and 1")
            self.translate = translate

            self.scale = scale

            if shear is not None:
                if isinstance(shear, numbers.Number):
                    if shear < 0:
                        raise ValueError("If shear is a single number, it must be positive.")
                    self.shear = (-shear, shear)
                else:
                    assert isinstance(shear, (tuple, list)) and len(shear) == 2
                    self.shear = shear
            else:
                self.shear = shear

            self.resample = resample
            self.fillcolor = fillcolor

        @staticmethod
        def get_params(degrees, translate, scale_ranges, shears, img_size):
            print(type(degrees), type(translate), type(scale_ranges), type(shears), type(img_size))
            print('-------')

            angle = degrees
            if translate is not None:
                max_dx = translate[0] * img_size[0]
                max_dy = translate[1] * img_size[1]
                translations = (np.round(max_dx),
                                np.round(max_dy))
            else:
                translations = (0, 0)

            if scale_ranges is not None:
                scale = scale_ranges
            else:
                scale = 1.0

            if shears is not None:
                shear = random.uniform(shears[0], shears[1])
            else:
                shear = 0.0

            return angle, translations, scale, shear

        def __call__(self, img):
            ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
            return F.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor)

        def __repr__(self):
            s = '{name}(degrees={degrees}'
            if self.translate is not None:
                s += ', translate={translate}'
            if self.scale is not None:
                s += ', scale={scale}'
            if self.shear is not None:
                s += ', shear={shear}'
            if self.resample > 0:
                s += ', resample={resample}'
            if self.fillcolor != 0:
                s += ', fillcolor={fillcolor}'
            s += ')'
            d = dict(self.__dict__)
            d['resample'] = _pil_interpolation_to_str[d['resample']]
            return s.format(name=self.__class__.__name__, **d)

class ToTensor(object):

    def __call__(self, pic):
        return F.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToPILImage(object):

    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic):
        return F.to_pil_image(pic, self.mode)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.mode is not None:
            format_string += 'mode={0}'.format(self.mode)
        format_string += ')'
        return format_string


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return F.normalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Resize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return F.resize(img, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class Scale(Resize):

    def __init__(self, *args, **kwargs):
        warnings.warn("The use of the transforms.Scale transform is deprecated, " +
                      "please use transforms.Resize instead.")
        super(Scale, self).__init__(*args, **kwargs)


class CenterCrop(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        return F.center_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class Pad(object):

    def __init__(self, padding, fill=0, padding_mode='constant'):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " + "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        return F.pad(img, self.padding, self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.padding, self.fill, self.padding_mode)


class Lambda(object):

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomTransforms(object):

    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomApply(RandomTransforms):

    def __init__(self, transforms, p=0.5):
        super(RandomApply, self).__init__(transforms)
        self.p = p

    def __call__(self, img):
        if self.p < random.random():
            return img
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomOrder(RandomTransforms):

    def __call__(self, img):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            img = self.transforms[i](img)
        return img


class RandomChoice(RandomTransforms):

    def __call__(self, img):
        t = random.choice(self.transforms)
        return t(img)


class RandomCrop(object):

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):

        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):

        if self.padding > 0:
            img = F.pad(img, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):

        if random.random() < self.p:
            return F.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return F.vflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string
class RandomSizedCrop(RandomResizedCrop):
    def __init__(self, *args, **kwargs):
        warnings.warn("The use of the transforms.RandomSizedCrop transform is deprecated, " +
                      "please use transforms.RandomResizedCrop instead.")
        super(RandomSizedCrop, self).__init__(*args, **kwargs)
class FiveCrop(object):
    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return F.five_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
class TenCrop(object):
    def __init__(self, size, vertical_flip=False):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size
        self.vertical_flip = vertical_flip

    def __call__(self, img):
        return F.ten_crop(img, self.size, self.vertical_flip)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, vertical_flip={1})'.format(self.size, self.vertical_flip)
class LinearTransformation(object):

    def __init__(self, transformation_matrix):
        if transformation_matrix.size(0) != transformation_matrix.size(1):
            raise ValueError("transformation_matrix should be square. Got " +
                             "[{} x {}] rectangular matrix.".format(*transformation_matrix.size()))
        self.transformation_matrix = transformation_matrix

    def __call__(self, tensor):

        if tensor.size(0) * tensor.size(1) * tensor.size(2) != self.transformation_matrix.size(0):
            raise ValueError("tensor and transformation matrix have incompatible shape." +
                             "[{} x {} x {}] != ".format(*tensor.size()) +
                             "{}".format(self.transformation_matrix.size(0)))
        flat_tensor = tensor.view(1, -1)
        transformed_tensor = torch.mm(flat_tensor, self.transformation_matrix)
        tensor = transformed_tensor.view(tensor.size())
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += (str(self.transformation_matrix.numpy().tolist()) + ')')
        return format_string
class ColorJitter(object):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):

        transforms = []
        if brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):

        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string
class RandomRotation(object):

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):

        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img):
        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


# # class RandomAffine(object):
#     """Random
#     def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
#         if isinstance(degrees, numbers.Number):
#             if degrees < 0:
#                 raise ValueError("If degrees is a single number, it must be positive.")
#             self.degrees = (-degrees, degrees)
#         else:
#             assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
#                 "degrees should be a list or tuple and it must be of length 2."
#             self.degrees = degrees
#
#         if translate is not None:
#             assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
#                 "translate should be a list or tuple and it must be of length 2."
#             for t in translate:
#                 if not (0.0 <= t <= 1.0):
#                     raise ValueError("translation values should be between 0 and 1")
#         self.translate = translate
#
#         if scale is not None:
#             assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
#                 "scale should be a list or tuple and it must be of length 2."
#             for s in scale:
#                 if s <= 0:
#                     raise ValueError("scale values should be positive")
#         self.scale = scale
#
#         if shear is not None:
#             if isinstance(shear, numbers.Number):
#                 if shear < 0:
#                     raise ValueError("If shear is a single number, it must be positive.")
#                 self.shear = (-shear, shear)
#             else:
#                 assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
#                     "shear should be a list or tuple and it must be of length 2."
#                 self.shear = shear
#         else:
#             self.shear = shear
#
#         self.resample = resample
#         self.fillcolor = fillcolor
#
#     @staticmethod
#     def get_params(degrees, translate, scale_ranges, shears, img_size):
#         """Get parameters for affine transformation
#
#         Returns:
#             sequence: params to be passed to the affine transformation
#         """
#         angle = random.uniform(degrees[0], degrees[1])
#         if translate is not None:
#             max_dx = translate[0] * img_size[0]
#             max_dy = translate[1] * img_size[1]
#             translations = (np.round(random.uniform(-max_dx, max_dx)),
#                             np.round(random.uniform(-max_dy, max_dy)))
#         else:
#             translations = (0, 0)
#
#         if scale_ranges is not None:
#             scale = random.uniform(scale_ranges[0], scale_ranges[1])
#         else:
#             scale = 1.0
#
#         if shears is not None:
#             shear = random.uniform(shears[0], shears[1])
#         else:
#             shear = 0.0
#
#         return angle, translations, scale, shear
#
#     def __call__(self, img):
#         """
#             img (PIL Image): Image to be transformed.
#
#         Returns:
#             PIL Image: Affine transformed image.
#         """
#         ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
#         return F.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor)
#
#     def __repr__(self):
#         s = '{name}(degrees={degrees}'
#         if self.translate is not None:
#             s += ', translate={translate}'
#         if self.scale is not None:
#             s += ', scale={scale}'
#         if self.shear is not None:
#             s += ', shear={shear}'
#         if self.resample > 0:
#             s += ', resample={resample}'
#         if self.fillcolor != 0:
#             s += ', fillcolor={fillcolor}'
#         s += ')'
#         d = dict(self.__dict__)
#         d['resample'] = _pil_interpolation_to_str[d['resample']]
#         return s.format(name=self.__class__.__name__, **d)
#
#
# class Grayscale(object):
#     """Convert image to grayscale.
#
#     Args:
#         num_output_channels (int): (1 or 3) number of channels desired for output image
#
#     Returns:
#         PIL Image: Grayscale version of the input.
#         - If num_output_channels == 1 : returned image is single channel
#         - If num_output_channels == 3 : returned image is 3 channel with r == g == b
#
#     """
#
#     def __init__(self, num_output_channels=1):
#         self.num_output_channels = num_output_channels
#
#     def __call__(self, img):
#         """
#         Args:
#             img (PIL Image): Image to be converted to grayscale.
#
#         Returns:
#             PIL Image: Randomly grayscaled image.
#         """
#         return F.to_grayscale(img, num_output_channels=self.num_output_channels)
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(num_output_channels={0})'.format(self.num_output_channels)
#
# class RandomGrayscale(object):
#     """Randomly convert image to grayscale with a probability of p (default 0.1).
#
#     Args:
#         p (float): probability that image should be converted to grayscale.
#
#     Returns:
#         PIL Image: Grayscale version of the input image with probability p and unchanged
#         with probability (1-p).
#         - If input image is 1 channel: grayscale version is 1 channel
#         - If input image is 3 channel: grayscale version is 3 channel with r == g == b
#
#     """
#
#     def __init__(self, p=0.1):
#         self.p = p
#
#     def __call__(self, img):
#         """
#         Args:
#             img (PIL Image): Image to be converted to grayscale.
#
#         Returns:
#             PIL Image: Randomly grayscaled image.
#         """
#         num_output_channels = 1 if img.mode == 'L' else 3
#         if random.random() < self.p:
#             return F.to_grayscale(img, num_output_channels=num_output_channels)
#         return img
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(p={0})'.format(self.p)

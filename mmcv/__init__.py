import colorsys
import numpy as np
from PIL import Image


__all__ = ['imread', 'imnormalize', 'bgr2hsv', 'hsv2bgr']


def imread(filename, flag='unchanged'):
    img = Image.open(filename)
    if flag == 'grayscale':
        arr = np.array(img.convert('L'))
        return arr

    arr = np.array(img)
    if arr.ndim == 2:
        return arr

    if arr.shape[-1] == 3:
        arr = arr[..., ::-1]
    elif arr.shape[-1] == 4:
        arr = arr[..., [2, 1, 0, 3]]
    return arr


def imnormalize(img, mean, std, to_rgb=True):
    img = img.astype(np.float32)
    if to_rgb and img.ndim == 3 and img.shape[-1] >= 3:
        img = img.copy()
        img[..., :3] = img[..., :3][..., ::-1]
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return (img - mean) / std


def bgr2hsv(img):
    img = img.astype(np.float32)
    rgb = img[..., ::-1] / 255.0
    flat = rgb.reshape(-1, 3)
    hsv = np.empty_like(flat)
    for i, (r, g, b) in enumerate(flat):
        h, s, v = colorsys.rgb_to_hsv(float(r), float(g), float(b))
        hsv[i] = [h * 360.0, s, v]
    return hsv.reshape(img.shape)


def hsv2bgr(img):
    hsv = img.astype(np.float32)
    flat = hsv.reshape(-1, 3)
    rgb = np.empty_like(flat)
    for i, (h, s, v) in enumerate(flat):
        r, g, b = colorsys.hsv_to_rgb(float(h) / 360.0, float(s), float(v))
        rgb[i] = [r, g, b]
    rgb = (rgb.reshape(img.shape) * 255.0)
    return rgb[..., ::-1]

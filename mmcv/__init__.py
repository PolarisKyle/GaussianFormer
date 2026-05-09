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
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    h = np.zeros_like(maxc)
    s = np.zeros_like(maxc)
    v = maxc

    nonzero = maxc > 1e-12
    s[nonzero] = delta[nonzero] / maxc[nonzero]

    nonzero_delta = delta > 1e-12
    r_mask = nonzero_delta & (maxc == r)
    g_mask = nonzero_delta & (maxc == g)
    b_mask = nonzero_delta & (maxc == b)

    h[r_mask] = ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6.0
    h[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2.0
    h[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4.0
    h = h * 60.0

    return np.stack([h, s, v], axis=-1)


def hsv2bgr(img):
    hsv = img.astype(np.float32)
    h = (hsv[..., 0] / 60.0) % 6.0
    s = hsv[..., 1]
    v = hsv[..., 2]

    c = v * s
    x = c * (1.0 - np.abs((h % 2.0) - 1.0))
    m = v - c

    z = np.zeros_like(c)
    rp = np.select(
        [h < 1, h < 2, h < 3, h < 4, h < 5, h <= 6],
        [c, x, z, z, x, c],
        default=z,
    )
    gp = np.select(
        [h < 1, h < 2, h < 3, h < 4, h < 5, h <= 6],
        [x, c, c, x, z, z],
        default=z,
    )
    bp = np.select(
        [h < 1, h < 2, h < 3, h < 4, h < 5, h <= 6],
        [z, z, x, c, c, x],
        default=z,
    )

    rgb = np.stack([rp + m, gp + m, bp + m], axis=-1) * 255.0
    return rgb[..., ::-1]

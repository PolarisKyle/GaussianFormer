import json
import os
import pickle

from .config import Config, ConfigDict
from .registry import Registry, MODELS, build_from_cfg
from .logging import MMLogger


__all__ = [
    'Config', 'ConfigDict', 'Registry', 'MODELS', 'build_from_cfg', 'MMLogger', 'load'
]


def load(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.pkl', '.pickle']:
        with open(path, 'rb') as f:
            return pickle.load(f)
    if ext == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    if ext in ['.npy', '.npz']:
        import numpy as np
        return np.load(path, allow_pickle=True)
    raise ValueError(f'Unsupported file format for load(): {ext}')

from mmseg.registry import MODELS

from . import builder
from .builder import build, build_backbone, build_neck, build_head, build_segmentor

SEGMENTORS = MODELS
HEADS = MODELS
BACKBONES = MODELS
NECKS = MODELS
LOSSES = MODELS

from .backbones import *
from .necks import *
from .losses import *

__all__ = [
    'MODELS',
    'SEGMENTORS',
    'HEADS',
    'BACKBONES',
    'NECKS',
    'LOSSES',
    'builder',
    'build',
    'build_backbone',
    'build_neck',
    'build_head',
    'build_segmentor',
]

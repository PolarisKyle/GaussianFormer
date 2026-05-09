from mmseg.registry import MODELS


def build(cfg):
    return MODELS.build(cfg)


def build_backbone(cfg):
    return MODELS.build(cfg)


def build_neck(cfg):
    return MODELS.build(cfg)


def build_head(cfg):
    return MODELS.build(cfg)


def build_segmentor(cfg):
    return MODELS.build(cfg)

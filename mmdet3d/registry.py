from mmengine.registry import Registry

MODELS = Registry('mmdet3d_models')

try:
    import mmdet3d.models  # noqa: F401
except Exception:
    pass

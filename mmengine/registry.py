from copy import deepcopy


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._module_dict = {}

    def get(self, key):
        return self._module_dict.get(key, None)

    def register_module(self, module=None, name=None, force=False):
        def _register(cls):
            module_name = name or cls.__name__
            if not force and module_name in self._module_dict:
                raise KeyError(f"{module_name} is already registered in {self.name}")
            self._module_dict[module_name] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register

    def build(self, cfg, default_args=None):
        return build_from_cfg(cfg, self, default_args=default_args)


MODELS = Registry('models')


def build_from_cfg(cfg, registry: Registry, default_args=None):
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')

    args = deepcopy(cfg)
    if default_args:
        for k, v in default_args.items():
            args.setdefault(k, v)

    obj_type = args.pop('type', None)
    if obj_type is None:
        raise KeyError('`type` is required in config to build module')

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f'{obj_type} is not in registry {registry.name}')
    elif isinstance(obj_type, type):
        obj_cls = obj_type
    else:
        raise TypeError(f'type must be str or class, but got {type(obj_type)}')

    return obj_cls(**args)

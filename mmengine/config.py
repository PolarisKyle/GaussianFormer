import os
import pprint


class ConfigDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        self[name] = _to_config_dict(value)

    def __delattr__(self, name):
        del self[name]


class Config(ConfigDict):
    def __init__(self, cfg_dict=None, filename=None):
        super().__init__()
        cfg_dict = cfg_dict or {}
        for k, v in cfg_dict.items():
            self[k] = _to_config_dict(v)
        self.filename = filename

    @classmethod
    def fromfile(cls, filename):
        filename = os.path.abspath(filename)
        cfg_dict = _load_py_config(filename)
        return cls(cfg_dict=cfg_dict, filename=filename)

    @property
    def pretty_text(self):
        return pprint.pformat(_to_plain_dict(self), sort_dicts=False, width=120)

    def dump(self, file):
        with open(file, 'w', encoding='utf-8') as f:
            f.write(self.pretty_text + '\n')


def _to_config_dict(value):
    if isinstance(value, ConfigDict):
        return value
    if isinstance(value, dict):
        out = ConfigDict()
        for k, v in value.items():
            out[k] = _to_config_dict(v)
        return out
    if isinstance(value, list):
        return [_to_config_dict(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_config_dict(v) for v in value)
    return value


def _to_plain_dict(value):
    if isinstance(value, dict):
        return {k: _to_plain_dict(v) for k, v in value.items() if k != 'filename'}
    if isinstance(value, list):
        return [_to_plain_dict(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_plain_dict(v) for v in value)
    return value


def _load_py_vars(filename):
    if not filename.endswith('.py'):
        raise ValueError('Only .py config files are supported.')
    cfg_globals = {'__file__': filename}
    with open(filename, 'r', encoding='utf-8') as f:
        # NOTE: Python config execution is intended only for trusted local config files.
        code = compile(f.read(), filename, 'exec')
        exec(code, cfg_globals)
    cfg = {}
    for k, v in cfg_globals.items():
        if k.startswith('__'):
            continue
        cfg[k] = v
    return cfg


def _deep_merge(base, child):
    if not isinstance(base, dict) or not isinstance(child, dict):
        return _to_config_dict(child)

    if child.get('_delete_', False):
        child = {k: v for k, v in child.items() if k != '_delete_'}
        return _to_config_dict(child)

    out = {k: _to_config_dict(v) for k, v in base.items()}
    for k, v in child.items():
        if k == '_delete_':
            continue
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = _to_config_dict(v)
    return _to_config_dict(out)


def _load_py_config(filename):
    cfg = _load_py_vars(filename)
    base_key = '_base_'
    if base_key not in cfg:
        return cfg

    base_files = cfg.pop(base_key)
    if isinstance(base_files, str):
        base_files = [base_files]

    merged = {}
    cur_dir = os.path.dirname(filename)
    for base_file in base_files:
        base_path = base_file if os.path.isabs(base_file) else os.path.join(cur_dir, base_file)
        base_cfg = _load_py_config(os.path.abspath(base_path))
        merged = _deep_merge(merged, base_cfg)

    merged = _deep_merge(merged, cfg)
    return merged

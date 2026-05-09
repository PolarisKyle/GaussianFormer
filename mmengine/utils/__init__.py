import os


class ManagerMixin:
    _instance_dict = {}

    def __init__(self, name='default'):
        self.name = name
        self.__class__._instance_dict[name] = self

    @classmethod
    def get_instance(cls, name='default'):
        return cls._instance_dict.get(name, None)


def symlink(src, dst):
    src = os.path.abspath(src)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)

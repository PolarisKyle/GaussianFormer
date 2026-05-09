import logging
import os


class MMLogger:
    _instance_dict = {}

    def __init__(self, name='mmengine', log_file=None, level=logging.INFO):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        if not self.logger.handlers:
            fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(fmt)
            self.logger.addHandler(stream_handler)

            if log_file:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(fmt)
                self.logger.addHandler(file_handler)

        MMLogger._instance_dict[name] = self

    @classmethod
    def get_instance(cls, name='mmengine', **kwargs):
        if name in cls._instance_dict:
            return cls._instance_dict[name]
        return cls(name=name, **kwargs)

    @classmethod
    def get_current_instance(cls):
        if cls._instance_dict:
            return next(reversed(cls._instance_dict.values()))
        return cls.get_instance('mmengine')

    def info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        self.logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        self.logger.error(*args, **kwargs)

    def debug(self, *args, **kwargs):
        self.logger.debug(*args, **kwargs)

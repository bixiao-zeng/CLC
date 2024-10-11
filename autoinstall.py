import sys
import os
from importlib import import_module


class AutoInstall():
    _loaded = set()

    @classmethod
    def find_spec(cls, name, path, target=None):
        if path is None and name not in cls._loaded:
            cls._loaded.add(name)
            print("Installing", name)
            try:
                result = os.system('pip install {}'.format(name))
                if result == 0:
                    return import_module(name)
            except Exception as e:
                print("Failed", e)
        return None


sys.meta_path.append(AutoInstall)

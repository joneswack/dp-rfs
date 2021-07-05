from collections.abc import Mapping 


class LazyDict(Mapping):
    """
    This dictionary assumes functions to be stored as values.
    These are then lazily evaluated.
    """
    def __init__(self, *args, **kw):
        self._raw_dict = dict(*args, **kw)

    def __getitem__(self, key):
        func = self._raw_dict.__getitem__(key)
        return func()

    def __iter__(self):
        return iter(self._raw_dict)

    def __len__(self):
        return len(self._raw_dict)
import numpy as np

# from https://docs.scipy.org/doc//numpy-1.11.0/user/basics.subclassing.html
class MetaInfoArray(np.ndarray):
    def __new__(cls, input_array, metainfo: dict):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.metainfo = metainfo
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # Called when the object is created from slicing or view
        if obj is None:
            return
        # Use getattr with default to avoid AttributeError
        self.metainfo = getattr(obj, 'metainfo', {})

class MetaInfoDict(dict):
    def __init__(self, *args, metainfo: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.metainfo = metainfo
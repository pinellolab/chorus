from typing import ClassVar, Any
from threading import RLock
import numpy as np
from scipy.interpolate import make_interp_spline
from typing import Callable

class Interpolation:
    _registry: ClassVar[dict[str, 'Interpolation']] = {}
    _lock: ClassVar[RLock] = RLock()
    
    @classmethod
    def from_string(cls, name: str):
        name = name.lower()
        return Interpolation._registry[name]()

    def __init_subclass__(cls, *, register: bool = True, name: str | None = None, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        if not register:
            return  # allow abstract/templates to skip registration

        if name is None:
            raise TypeError(
                f"{cls.__name__}: name must be passed using name argument when subclassing) for registration"
            )
        name = name.lower()
        cls.name = name 

        with Interpolation._lock:
            if name in Interpolation._registry and Interpolation._registry[name] is not cls:
                raise ValueError(
                    f"Duplicate registration for '{name}': "
                    f"{Interpolation._registry[name].__name__} already registered"
                )
            Interpolation._registry[name] = cls

    def fit(self, resolution: int, values: np.ndarray[np.float32]):
        raise NotImplementedError()

    def predict(self, resolution, interval_end):
        raise NotImplementedError()

    def interpolation_func(self, coords: np.ndarray[np.float32], values: np.ndarray[np.float32]) -> Callable[[np.ndarray[np.float32]], np.ndarray[np.float32]]:
        raise NotImplementedError()

class LinearInterpolation(Interpolation, name='linear'):
    '''
    simple linear interpolation
    '''
    def interpolation_func(self, coords: np.ndarray[np.float32], values: np.ndarray[np.float32]) -> Callable[[np.ndarray[np.float32]], np.ndarray[np.float32]]:
        return make_interp_spline(coords, values, k=1)

    def fit(self, resolution: int, values: np.ndarray[np.float32]):
        coords = np.arange(0, values.shape[0] ) * resolution + resolution // 2
        self.f = self.interpolation_func(coords, values)
        self.resolution = resolution
        return self

    def predict(self, resolution: int, interval_end: int):
        size, rest = divmod(interval_end, resolution)
        size = size + (rest != 0)
        coords = np.arange(0, size) * resolution + resolution // 2 
        values = self.f(coords)
        return values

class LinearDividedInterpolation(Interpolation, name='linear_divided'):
    '''
    simple linear interpolation, but divided by the resolution
    '''
    def interpolation_func(self, coords: np.ndarray[np.float32], values: np.ndarray[np.float32]) -> Callable[[np.ndarray[np.float32]], np.ndarray[np.float32]]:
        return make_interp_spline(coords, values, k=1)
    
    def fit(self, resolution: int, values: np.ndarray[np.float32]):
        coords = np.arange(0, values.shape[0] ) * resolution + resolution // 2
        self.f = self.interpolation_func(coords, values)
        self.resolution = resolution
        return self

    def predict(self, resolution: int, interval_end: int):
        size, rest = divmod(interval_end, resolution)
        size = size + (rest != 0)
        coords = np.arange(0, size) * resolution + resolution // 2 
        out_values = self.f(coords)
        scale_factor = resolution / self.resolution
        out_values = out_values * scale_factor
        return out_values

class CubicInterpolation(Interpolation, name='cubic'):
    '''
    spline cubic interpolation
    '''
    def interpolation_func(self, coords: np.ndarray[np.float32], values: np.ndarray[np.float32]) -> Callable[[np.ndarray[np.float32]], np.ndarray[np.float32]]:
        return make_interp_spline(coords, values, k=3)
    
    def fit(self, resolution: int, values: np.ndarray[np.float32]):
        coords = np.arange(0, values.shape[0] ) * resolution + resolution // 2
        self.f = self.interpolation_func(coords, values)
        self.resolution = resolution
        return self

    def predict(self, resolution, interval_end: int):
        size, rest = divmod(interval_end, resolution)
        size = size + (rest != 0)
        coords = np.arange(0, size) * resolution + resolution // 2 
        values = self.f(coords)
        return values


class BinInterpolation(Interpolation, name='bin'):
    '''
    just map each position to the nearest bin from the left
    '''
    def interpolation_func(self, coords: np.ndarray[np.float32], values: np.ndarray[np.float32]) -> Callable[[np.ndarray[np.float32]], np.ndarray[np.float32]]:
        return make_interp_spline(coords, values, k=0)
    
    def fit(self, resolution: int, values: np.ndarray[np.float32]):
        coords = np.arange(0, values.shape[0] ) * resolution 
        self.f = self.interpolation_func(coords, values)
        self.resolution = resolution
        return self

    def predict(self, resolution: int, interval_end: int):
        size, rest = divmod(interval_end, resolution)
        size = size + (rest != 0)
        coords = np.arange(0, size) * resolution
        values = self.f(coords)
        return values
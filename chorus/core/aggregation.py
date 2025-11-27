from typing import ClassVar, Any, Type
from threading import RLock
import numpy as np

from ..core.interpolation import Interpolation
from typing import Type

class Aggregation:
    _registry: ClassVar[dict[str, 'Aggregation']] = {}
    _lock: ClassVar[RLock] = RLock()
    
    @classmethod
    def from_string(cls, name: str):
        name = name.lower()
        return Aggregation._registry[name]()

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

        with Aggregation._lock:
            if name in Aggregation._registry and Aggregation._registry[name] is not cls:
                raise ValueError(
                    f"Duplicate registration for '{name}': "
                    f"{Interpolation._registry[name].__name__} already registered"
                )
            Aggregation._registry[name] = cls

    def aggregate(self, 
                  values: np.ndarray[np.float32],
                  resolution: int, 
                  new_resolutions: int, 
                  interpolation_type: str | Type[Interpolation] = 'linear_divided'):
        raise NotImplementedError()


class PoolingAggregations(Aggregation, register=False): # intermediate class
    def aggregation_func(self, vals: np.ndarray[np.float32]) -> np.float32:
        raise NotImplementedError('Aggregation function must be specified')
    
    def aggregate(self, 
                  values: np.ndarray[np.float32],
                  resolution: int, 
                  new_resolution: int, 
                  interpolation: str | Type[Interpolation] = 'linear_divided'):
        if isinstance(interpolation, str):
            interp = Interpolation.from_string(interpolation) 
        else:
            interp = interpolation()
        
        interval_end = values.shape[0] * resolution
        intermediate_resolution = np.gcd(resolution, new_resolution)
        if intermediate_resolution == resolution: #  target resolution is not an exact multiple of the initial one
            inter_values = values
        else:
            interp = interpolation()
            interp.fit(resolution=resolution, values=values)
            
            inter_values = interp.predict(resolution=intermediate_resolution, 
                                          interval_end= interval_end)

        relative_step = new_resolution // intermediate_resolution
        out_size, rest = divmod(interval_end, new_resolution)
        out_size = out_size + (rest != 0)

        out_vals = np.zeros(out_size, dtype=np.float32)
        for ind in range(out_size):
            start = ind * relative_step
            end = start + relative_step 
            val = self.aggregation_func(inter_values[start:end])
            out_vals[ind] = val
        return out_vals 

class SumAggregation(PoolingAggregations, name='sum'):
    def aggregation_func(self, vals: np.ndarray[np.float32]) -> np.float32:
        return np.sum(vals)

class MeanAggregation(PoolingAggregations, name='mean'):
    def aggregation_func(self, vals: np.ndarray[np.float32]) -> np.float32:
        return np.mean(vals)

class MedianAggregation(PoolingAggregations, name='median'):
    def aggregation_func(self, vals: np.ndarray[np.float32]) -> np.float32:
        return np.median(vals)

class MaxAggregation(PoolingAggregations, name='max'):
    def aggregation_func(self, vals: np.ndarray[np.float32]) -> np.float32:
        return np.max(vals)

class MinAggregation(PoolingAggregations, name='min'):
    def aggregation_func(self, vals: np.ndarray[np.float32]) -> np.float32:
        return np.min(vals)
import numpy as np
from typing import Union

class Model:
    """Parent Object that prepare basic functions"""
    def __init__(self):
        self.has_fit = False
    
    def _dependance_check(fn):
        def wrapper(self, *args, **kwargs):
            if not getattr(self, "has_fit", False):
                raise ValueError("Attempted prediction without model fit")
            return fn(self, *args, **kwargs)
        return wrapper

    def fit(self, x:Union[list, np.ndarray], y:Union[list, np.ndarray]):
        """
        Train the model to the given data
        
        Parameters
        ----------
        x:list or numpy.ndarray
        y:list or numpy.ndarray
        """
        if type(x) is list:
            x = np.array(x)
        if type(y) is list:
            y = np.array(y)
        
        if y.shape[0] != x.shape[0]:
            raise ValueError(f"The length of samples is not the same in {x} and {y}")
        
        self._fit(x, y)
        
        self.has_fit = True 
        
    def _fit(self, x:np.ndarray, y:np.ndarray):
        pass
    
    @_dependance_check
    def predict(self, x:Union[list, np.ndarray]) -> np.ndarray:
        """
        Return the predicted value based on the given inputs

        Parameters
        ----------
        x:list or numpy.ndarray

        Returns
        -------
        The list of predictions made
        """
        if type(x) is list:
            x = np.array(x)
        
        if len(x.shape) != 1:
            if x.shape[1] != self.x.shape[1]:
                raise ValueError("Not a valid numbers of features")
        
        return self._predict(x)
    
    def _predict(self, x:np.ndarray) -> np.ndarray:
        pass
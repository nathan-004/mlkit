import numpy as np
from typing import Union

class Model:
    """Parent Object that prepare basic functions"""
    
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
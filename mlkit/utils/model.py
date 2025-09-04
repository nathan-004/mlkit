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
        
        pass
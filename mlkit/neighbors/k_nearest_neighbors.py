import numpy as np
from typing import NamedTuple

from mlkit.utils.model import Model

class Point(NamedTuple):
    pass

class KNeighborsClassifier(Model):
    """Use of the KNN algorithm to solve classification problems"""
    
    def __init__(self, n_neighbors:int = 5):
        """
        Parameters
        ----------
        n_neighbors:int (default is 5)
            predict the next class by looking at the `n_neighbors` neighbors
        """
        super().__init__()
        self.n_neighbors = n_neighbors
        
    def _fit(self, x:np.ndarray, y:np.ndarray):
        """
        Prepare model to predict class by present neighbors
        Stock the points
        """

def main():
    a = KNeighborsClassifier()
    a.fit([1,2], [1,2])
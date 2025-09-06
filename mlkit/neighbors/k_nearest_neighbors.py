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
        self.x = x
        self.y = y

    @Model._dependance_check
    def _predict(self, x:np.ndarray) -> np.ndarray:
        """Find the n_nearest neighbors and return the class of most common"""
        distances = np.linalg.norm(self.x - x, axis=1)
        sorted_distances = np.sort(distances)

def main():
    a = KNeighborsClassifier(n_neighbors=3)
    a.fit([[0], [3], [2], [3]], [0,0,1,1])
    print(a.predict([[1.1]]))
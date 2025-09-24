import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import rainbow

from typing import NamedTuple
from collections import defaultdict

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
        classes = defaultdict(lambda : 0) # Contient la classe et le nombre de fois où c'est contenu
        total = 0
        for el in sorted_distances[:self.n_neighbors]:
            item_idx = np.where(distances == el)[0]
            for idx in item_idx:
                if total >= self.n_neighbors:
                    break
                classes[self.y[idx]] += 1
                total += 1
        return max(classes, key=lambda x: classes[x])
    
    @Model._dependance_check
    def plot(self, title="Scatter plot", xlabel="Feature 1", ylabel="Feature 2"):
        """Plot a scatter of the data, colored by class if available (max 2 features)."""
        
        # Vérifie que les données ont au moins 2 features
        if self.x.shape[1] < 2:
            raise ValueError("Les données doivent avoir au moins 2 features pour un scatter plot.")
        
        if hasattr(self, "y") and self.y is not None:
            classes = set(self.y)
            for cls in classes:
                mask = self.y == cls
                plt.scatter(self.x[mask, 0], self.x[mask, 1], label=f"Classe {cls}", alpha=0.7)
            plt.legend()
        else:
            plt.scatter(self.x[:, 0], self.x[:, 1], alpha=0.7)
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

def main():
    a = KNeighborsClassifier(n_neighbors=3)
    a.fit([[0,1], [1,2], [2,3], [3,4]], [0,0,1,1])
    print(a.predict([[1.1, 1.5]]))
    print(a.plot())

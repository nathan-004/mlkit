import numpy as np
import matplotlib.pyplot as plt

from typing import NamedTuple
from collections import defaultdict

from mlkit.utils.model import Model
from mlkit.datasets.classification import make_blobs

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

    def _single_predict(self, x:np.ndarray) -> np.ndarray:
        """Find the n_nearest neighbors and return the class of most common -> Returns the solution for one point"""
        distances = np.linalg.norm(self.x - x, axis=1)
        sorted_distances = np.sort(distances)
        classes = defaultdict(lambda : 0)
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
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit la classe pour un ou plusieurs points.
        X peut être un seul point (1D) ou un tableau de points (2D).
        """
        if X.ndim == 1:
            return self._single_predict(X)

        return np.array([self._single_predict(x) for x in X])
    
    @Model._dependance_check
    def plot(self, title="Scatter plot", xlabel="Feature 1", ylabel="Feature 2"):
        """Plot a scatter of the data, colored by class if available (max 2 features)."""
        
        # Vérifie que les données ont au moins 2 features
        if self.x.shape[1] != 2:
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
        plt.savefig('figures/scatter.png')
        plt.show()

    def plot_predict(self, step=0.1, title="Zones de décision prédites", xlabel="Feature 1", ylabel="Feature 2"):
        """Display the zones based on the predictions of the model (Only 2D)."""

        if self.x.shape[1] != 2:
            raise ValueError("plot_predict ne fonctionne qu'avec 2 features.")

        # Définir les limites du plan
        x_min, x_max = self.x[:, 0].min() - 1, self.x[:, 0].max() + 1
        y_min, y_max = self.x[:, 1].min() - 1, self.x[:, 1].max() + 1

        # Créer une grille de points
        xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                            np.arange(y_min, y_max, step))

        # Prédire la classe pour chaque point de la grille
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid_points)
        Z = Z.reshape(xx.shape)

        # Afficher la zone colorée selon les classes
        plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")

        if hasattr(self, "y") and self.y is not None:
            plt.scatter(self.x[:, 0], self.x[:, 1], c=self.y, cmap="coolwarm", edgecolors="k")
        else:
            plt.scatter(self.x[:, 0], self.x[:, 1], edgecolors="k")

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig('figures/decision_sone.png')
        plt.show()

def main():
    X, y = make_blobs(n_classes=4, n_samples=100)
    a = KNeighborsClassifier(n_neighbors=10)
    a.fit(X, y)
    a.plot()
    a.plot_predict(step=0.01)

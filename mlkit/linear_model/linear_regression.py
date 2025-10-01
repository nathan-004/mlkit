import numpy as np
import matplotlib.pyplot as plt
import random

from mlkit.utils.model import Model
from mlkit.linear_model.error_functions import MeanSquaredError, MeanAbsoluteError
from mlkit.datasets.continuous import linear_dataset

class LinearRegression(Model):
    """Solve Linear Problems with LinearRegression"""

    def __init__(self):
        """
        Parameters
        ----------
        To Complete
        """
        super().__init__()
    
    def _fit(self, x:np.ndarray, y:np.ndarray):
        """
        Find the linear function of the datas
        Predict continuous values
        
        Parameters
        ----------
        x:np.ndarray
            Inputs datas in 1D array
        y:np.ndarray
            Result points in 1D array
        """
        assert len(x.shape) == 1, "x must be a 1D array"
        assert len(y.shape) == 1, "y must be a 1D array"
        assert x.shape[0] == y.shape[0], "x and y must have the same length"

        self.x = x
        self.y = y

        self.a, self.b = self._least_squares(x, y)

    def _least_squares(self, x:np.ndarray, y:np.ndarray) -> tuple:
        """
        Find the linear function of the datas using least squares method
        
        Parameters
        ----------
        x:np.ndarray
            Inputs datas in 1D array
        y:np.ndarray
            Result points in 1D array

        Returns
        -------
        a:float
            Slope of the linear function
        b:float
            Intercept of the linear function
        """
        n = len(x)
        a = (n * np.dot(x, y) - np.sum(x) * np.sum(y)) / (n * np.dot(x, x) - (np.sum(x))**2)
        b = (np.sum(y) - a * np.sum(x)) / n
        return a, b
    
    def _predict(self, x:np.ndarray) -> np.ndarray:
        """
        Predict the result points using the linear function
        
        Parameters
        ----------
        x:np.ndarray
            Inputs datas in 1D array

        Returns
        -------
        y_pred:np.ndarray
            Predicted result points in 1D array
        """
        assert len(x.shape) == 1, "x must be a 1D array"
        return self.a * x + self.b
    
    def plot(self):
        """
        Plot the datas and the linear function
        """
        plt.scatter(self.x, self.y, color='blue', label='Data points')
        plt.plot(self.x, self._predict(self.x), color='red', label='Fitted line')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linear Regression Fit')
        plt.legend()
        plt.show()
        plt.savefig("figures/linear_regression.png")

def main():
    # Example usage
    x, y = linear_dataset(start=0, end=50, xstep=0.5, a=2, b=3, ynoise=20, seed=42)

    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    print("Predicted values:", y_pred)

    model.plot()
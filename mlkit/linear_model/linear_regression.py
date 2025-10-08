import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

from mlkit.utils.model import Model
from mlkit.linear_model.error_functions import MeanSquaredError, MeanAbsoluteError, ErrorFunction
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

        self.a, self.b = self._minimizing_error(x, y)

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

    def _minimizing_error(self, x:np.ndarray, y:np.ndarray, error_function:ErrorFunction = MeanSquaredError(), animation:bool = True) -> tuple:
        """
        Minimize the error to find the best fitted line

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
        a, b = 1.0, 0.0
        lr = 0.001  # taux d'apprentissage
        a_values = [a]
        b_values = [b]

        for epoch in range(1000):
            y_pred = a * x + b
            grad_a = np.mean(error_function.derivative(y, y_pred) * x)
            grad_b = np.mean(error_function.derivative(y, y_pred))
            a -= lr * grad_a
            b -= lr * grad_b
            a_values.append(a)
            b_values.append(b)

        if animation:
            self._animate(a_values, b_values,x, y)

        return a, b
    
    def _animate(self, a_values:np.ndarray, b_values:np.ndarray, x:np.ndarray, y:np.ndarray):
        fig, ax = plt.subplots()
        ax.scatter(x, y, color='blue')
        line, = ax.plot([], [], color='red', linewidth=2)
        ax.set_title("Descente de gradient")

        def update(frame):
            y_pred = a_values[frame] * x + b_values[frame]
            line.set_data(x, y_pred)
            ax.set_title(f"Epoch {frame}: a={a_values[frame]:.2f}, b={b_values[frame]:.2f}")
            return line,

        ani = FuncAnimation(fig, update, frames=len(a_values), interval=10, blit=True)
        plt.show()

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
    x, y = linear_dataset(start=0, end=50, xstep=0.5, a=-2, b=3, ynoise=20, seed=42)

    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    print("Predicted values:", y_pred)


    model.plot()

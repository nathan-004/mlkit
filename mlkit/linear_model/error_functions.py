import numpy as np
import matplotlib.pyplot as plt

class ErrorFunction:
    def __init__(self):
        pass

    def compute(self, y_true, y_pred):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def derivative(self, y_true, y_pred):
        raise NotImplementedError("Subclasses should implement this method.")

class MeanSquaredError(ErrorFunction):
    def function(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return ((y_true - y_pred) ** 2) / y_true.shape[0]

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        n = y_true.shape[0]
        return (2 / n) * (y_pred - y_true)

import numpy as np

class MeanAbsoluteError:
    def function(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.abs(y_true - y_pred)

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.where(y_pred > y_true, 1, np.where(y_pred < y_true, -1, 0))


def plot_function(error_function: ErrorFunction, y_true=1.0, y_pred_range=(0, 2), num_points=100):
    """Plot the error function in blue and its derivative in red."""
    y_pred = np.linspace(y_pred_range[0], y_pred_range[1], num_points)
    y_true_arr = np.full_like(y_pred, y_true)
    errors = error_function.function(y_true_arr, y_pred)

    derivatives = error_function.derivative(y_true_arr, y_pred)

    plt.plot(y_pred, errors, color='blue', label='Error Function')
    plt.plot(y_pred, derivatives, color='red', label='Derivative')
    plt.xlabel('y_pred')
    plt.ylabel('Value')
    plt.title(f'{error_function.__class__.__name__}')
    plt.legend()
    plt.savefig("figures/error_function.png")
    plt.show()

if __name__ == "__main__":
    plot_function(MeanAbsoluteError())
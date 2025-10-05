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
    def function(self, y_true:np.ndarray, y_pred:np.ndarray) -> np.ndarray:
        return (y_true - y_pred) ** 2
    def compute(self, y_true:np.ndarray, y_pred:np.ndarray) -> float:
        return ((y_true - y_pred) ** 2).mean()

class MeanAbsoluteError(ErrorFunction):
    def function(self, y_true:np.ndarray, y_pred:np.ndarray) -> np.ndarray:
        return np.abs(y_true - y_pred)
    def compute(self, y_true:np.ndarray, y_pred:np.ndarray) -> float:
        return np.abs(y_true - y_pred).mean()

def plot_function(error_function: ErrorFunction, y_true=1.0, y_pred_range=(0, 2), num_points=100):
    """Plot the error function in blue and its derivative in red."""
    y_pred = np.linspace(y_pred_range[0], y_pred_range[1], num_points)
    y_true_arr = np.full_like(y_pred, y_true)
    errors = error_function.function(y_true_arr, y_pred)
    print(errors, y_pred)
    plt.plot(y_pred, errors, color='blue', label='Error Function')
    plt.xlabel('y_pred')
    plt.title(f'{error_function.__class__.__name__}')
    plt.legend()
    plt.savefig("figures/error_function.png")
    plt.show()

if __name__ == "__main__":
    a = np.array([1])
    print(a)

    b = np.array([0])

    print(MeanSquaredError().compute(a, b))

    plot_function(MeanSquaredError())
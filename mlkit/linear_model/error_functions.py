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
    def compute(self, y_true:np.ndarray, y_pred:np.ndarray) -> float:
        return ((y_true - y_pred) ** 2).mean()

class MeanAbsoluteError(ErrorFunction):
    def compute(self, y_true:np.ndarray, y_pred:np.ndarray) -> float:
        return np.abs(y_true - y_pred).mean()

def plot_function(error_function:ErrorFunction):
    """Plot the function in blue and its derivative in red"""
    x = np.linspace(1, 100)

    plt.plot(x, color='blue', label='Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('')
    plt.legend()
    plt.show()
    plt.savefig("figures/error_function.png")

if __name__ == "__main__":
    a = np.array([1])
    print(a)

    b = np.array([0])

    print(MeanSquaredError().compute(a, b))

    plot_function(MeanAbsoluteError)
import numpy as np
import matplotlib.pyplot as plt
from mlkit.datasets.continuous import polymonial_dataset

def get_degree(x:np.ndarray, y:np.ndarray) -> int:
    """Return the degree of the function represented in the point set"""
    plt.scatter(x, y, color='blue', label='Data points')
    val = np.mean(y)
    plt.plot(x, [val for _ in range(len(x))], color='red', label='Fitted line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Polynomial Regression Fit')
    plt.legend()
    plt.savefig("figures/degree_calcul.png")
    plt.show()
    
def main():
    x, y = polymonial_dataset(-10000, 10000, 0.5, coeffs=[2,2,-2, -1, -1], seed=42)
    print(get_degree(x, y))
import numpy as np

class ErrorFunction:
    def __init__(self):
        pass

    def compute(self, y_true, y_pred):
        raise NotImplementedError("Subclasses should implement this method.")
    
class MeanSquaredError(ErrorFunction):
    def compute(self, y_true:np.ndarray, y_pred:np.ndarray) -> float:
        return ((y_true - y_pred) ** 2).mean()

class MeanAbsoluteError(ErrorFunction):
    def compute(self, y_true:np.ndarray, y_pred:np.ndarray) -> float:
        return np.abs(y_true - y_pred).mean()
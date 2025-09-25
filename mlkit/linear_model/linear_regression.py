import numpy as np

from mlkit.utils.model import Model

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
        
        Parameters
        ----------
        x:np.ndarray
            Inputs datas
        y:np.ndarray
            Result points
        """
        self.x = x
        self.y = y
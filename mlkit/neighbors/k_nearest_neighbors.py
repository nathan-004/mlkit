from mlkit.utils.model import Model

class KNeighborsClassifier(Model):
    """Use of the KNN algorithm to solve classification problems"""
    
    def __init__(self, n_neighbors:int = 5):
        """
        Parameters
        ----------
        n_neighbors:int (default is 5)
            predict the next class by looking at the `n_neighbors` neighbors
        """
        self.n_neighbors = n_neighbors

def main():
    pass
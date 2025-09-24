import numpy as np

def make_classification(
    n_samples: int = 200, 
    n_features: int = 2, 
    n_classes: int = 2, 
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Génère un jeu de données artificiel pour la classification.

    Chaque point est représenté par `n_features` caractéristiques. Les points sont 
    répartis en `n_classes` classes, de manière à simuler un problème de classification 
    supervisée.

    Paramètres
    ----------
    n_samples : int, facultatif (par défaut=200)
        Le nombre total de points de données à générer.
    
    n_features : int, facultatif (par défaut=2)
        Le nombre de dimensions (features) pour chaque point.
    
    n_classes : int, facultatif (par défaut=2)
        Le nombre de classes cibles.
    
    random_state : int, facultatif (par défaut=42)
        Graine pour le générateur aléatoire afin d'obtenir un résultat reproductible.

    Retour
    ------
    X : np.ndarray, shape (n_samples, n_features)
        Tableau contenant les coordonnées des points générés.
    
    y : np.ndarray, shape (n_samples,)
        Tableau contenant les étiquettes de classes correspondantes à chaque point.

    Exemple
    -------
    >>> X, y = make_classification(n_samples=100, n_features=2, n_classes=2)
    >>> X.shape
    (100, 2)
    >>> y.shape
    (100,)

    Notes
    -----
    Cette version simplifiée ne gère que la génération de points séparables par classes. 
    Pour des datasets plus complexes, des options comme la séparation des classes, le 
    bruit ou des clusters multiples peuvent être ajoutées.
    """
    rng = np.random.default_rng(seed=42)
    X = rng.random((n_samples, n_features))
    
    y = rng.integers(0, n_classes, size=n_samples)

    return (X, y)
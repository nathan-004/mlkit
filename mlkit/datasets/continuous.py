"""Generate random continuous datasets"""

from random import randint

def linear_dataset(start:float = 0, end:float = 10, xstep:int = 1, a:float = 1, b:float = 1, ynoise:int = 3):
    """
    Generate a random dataset around the linear function $ax + b$ with a random noise 
    """
    x = [i for i in range(start, end, xstep)]
    y = [a * i + b + randint(-ynoise, ynoise) for i in x]
    return (x, y)
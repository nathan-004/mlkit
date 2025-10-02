import inspect
from collections.abc import Callable

def get_calc(f:Callable[[float,], float]) -> Callable[[float,], float]:
    source_code = inspect.getsource(f)

    calc = source_code.strip().split("\n")[-1].strip()
    if calc.startswith("return "):
        calc = calc[len("return "):]

    return calc

def f(x):
    return x ** 2 + 5

if __name__ == "__main__":
    f_prime = get_calc(f)
    print(f_prime)
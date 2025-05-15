import math

from solver_utils import pretty_time

invphi = (math.sqrt(5) - 1) / 2  # 1 / phi

def gss(f, a, b, tolerance=1e-5):
    """
    Golden-section search
    to find the minimum of f on [a,b]

    * f: a strictly unimodal function on [a,b]

    Example:
    >>> def f(x): return (x - 2) ** 2
    >>> x = gss(f, 1, 5)
    >>> print(f"{x:.5f}")
    2.00000

    """
    while b - a > tolerance:
        print(f"Current interval: {pretty_time(a)}-{pretty_time(b)}")
        c = b - (b - a) * invphi
        d = a + (b - a) * invphi
        if f(c) < f(d):
            b = d
        else:  # f(c) > f(d) to find the maximum
            a = c

    return (b + a) / 2
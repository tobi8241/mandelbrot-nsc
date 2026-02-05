""""
Mandelbrot Set Generator
Author : [ Tobias Jagd ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import matplotlib.pyplot as plt

def mandelbrot_point (c, max_iter):
    """Calculate the number of iterations for a point in the Mandelbrot set."""
    z = 0
    for n in range(max_iter):
        z = z*z + c
        if abs(z) > 2:
            return n
    return max_iter

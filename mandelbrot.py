""""
Mandelbrot Set Generator
Author : [ Tobias Jagd ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import matplotlib.pyplot as plt
import time

def mandelbrot_point (c, max_iter):
    """Calculate the number of iterations for a point in the Mandelbrot set."""
    z = 0
    for n in range(max_iter):
        z = z*z + c
        if abs(z) > 2:
            return n
    return max_iter

def compute_mandelbrot_grid(xmin, xmax, ymin, ymax, width, height, max_iter):
    """Compute the mandelbrot grid for given region"""
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)

    counts = np.zeros((height, width), dtype=int)

    # loop through each point
    for j in range(height):
        for i in range(width):
            c = complex(x[i], y[j])
            counts[j, i] = mandelbrot_point(c, max_iter)
    return counts

if __name__ == "__main__":
    print(mandelbrot_point(0,100))

    start = time.time()
    grid = compute_mandelbrot_grid(-2, 1, -1.5, 1.5, 1024, 1024, 100)
    elapsed = time.time() - start

    print(grid)
    print(f"Computation took {elapsed:.3f} seconds")

    #plot
    plt.imshow(grid, cmap="hot", origin="lower")
    plt.colorbar(label="Iteration count")
    plt.title("Mandelbrot Set (naive)")
    plt.savefig("mandelbrot_naive.png")
    plt.show() 
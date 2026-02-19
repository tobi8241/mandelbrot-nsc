""""
Mandelbrot Set Generator
Author : [ Tobias Jagd ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import matplotlib.pyplot as plt
import time , statistics


def benchmark (func, *args, n_runs =3) :
    """ Time func , return median of n_runs . """
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)
    median_t = statistics.median(times)
    print(f" Median: {median_t:.4f}s "
           f"( min={min(times):.4f}, max ={max(times):.4f})")
    return median_t , result



def mandelbrot_point (c, max_iter):
    """Calculate the number of iterations for a point in the Mandelbrot set."""
    z = 0
    for n in range(max_iter):
        z = z*z + c
        if abs(z) > 2:
            return n
    return max_iter

def compute_naive_mandelbrot_grid(xmin, xmax, ymin, ymax, width, height, max_iter):
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

def compute_numpy_mandelbrot_grid(xmin, xmax, ymin, ymax, width, height, max_iter):
    """Vectorized Mandelbrot using NumPy arrays."""

    # Create complex grid 
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    
    Z = np.zeros_like(C, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=int)

    for _ in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        M[mask] += 1
    return M
        

if __name__ == "__main__":

    start = time.time()
    grid = compute_numpy_mandelbrot_grid(-2, 1, -1.5, 1.5, 1024, 1024, 100)
    elapsed = time.time() - start

    print(grid)
    print(f"Computation took {elapsed:.3f} seconds")

    t , M = benchmark ( compute_numpy_mandelbrot_grid , -2, 1, -1.5 , 1.5 , 1024 , 1024 , 100)

    #plot
    plt.imshow(grid, cmap="hot", origin="lower")
    plt.colorbar(label="Iteration count")
    plt.title("Mandelbrot Set (numpy)")
    plt.savefig("mandelbrot_numpy.png")
    plt.show() 
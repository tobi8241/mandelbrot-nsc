""""
Mandelbrot Set Generator
Author : [ Tobias Jagd ]
Course : Numerical Scientific Computing 2026
"""
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import time , statistics 

def row_sums(A):
        N = A.shape[0]
        s = 0.0
        for i in range(N):
            s += np.sum(A[i, :])
        return s
    
def col_sums(A):
        N = A.shape[1]
        s = 0.0
        for j in range(N):
            s += np.sum(A[:, j])
        return s

def benchmark (func, *args, n_runs =5):
    """ Time func , return median of n_runs . """
    func(*args)  # warmup
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


"""
def mandelbrot_point (c, max_iter):
    #Calculate the number of iterations for a point in the Mandelbrot set.
    z = 0
    for n in range(max_iter):
        if abs(z) >= 2:
            return n
        z = z*z + c
    return max_iter

def compute_naive_mandelbrot_grid(xmin, xmax, ymin, ymax, width, height, max_iter):
    #Compute the mandelbrot grid for given region
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)

    counts = np.zeros((height, width), dtype=int)

    # loop through each point
    for j in range(height):
        for i in range(width):
            c = complex(x[i], y[j])
            counts[j, i] = mandelbrot_point(c, max_iter)
    return counts
"""

def mandelbrot_naive(xmin, xmax, ymin, ymax, width, height, max_iter=100):
     x = np.linspace(xmin, xmax, width)
     y = np.linspace(ymin, ymax, height)
     results = np.zeros((height, width), dtype=int)
     for i in range(height):
         for j in range(width):
             c = x[j] + 1j * y[i]
             z = 0
             for n in range(max_iter):
                 if abs(z) > 2:
                     results[i, j] = n
                     break
                 z = z*z + c
             else:
                results[i, j] = max_iter
     return results

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

@njit
def mandelbrot_point_numba(c, max_iter=100):
     z = 0j
     for n in range(max_iter):
          if z.real*z.real + z.imag*z.imag > 4.0:
               return n
          z = z*z + c
     return max_iter

def mandelbrot_hybrid(xmin, xmax, ymin, ymax, width, height, max_iter=100):
     x = np.linspace(xmin, xmax, width)
     y = np.linspace(ymin, ymax, height)
     results = np.zeros((height, width), dtype=np.int32)
     for i in range(height):
         for j in range(width):
             c = x[j] + 1j * y[i]
             results[i, j] = mandelbrot_point_numba(c, max_iter)
     return results

@njit
def mandelbrot_naive_numba(xmin, xmax, ymin, ymax, width, height, max_iter=100):
     x = np.linspace(xmin, xmax, width)
     y = np.linspace(ymin, ymax, height)
     results = np.zeros((height, width), dtype=np.int32)
     for i in range(height):
         for j in range(width):
             c = x[j] + 1j * y[i]
             z = 0j
             n = 0
             while n < max_iter and (z.real*z.real + z.imag*z.imag) <= 4.0:
                 z = z*z + c
                 n += 1
             results[i, j] = n
     return results

@njit
def mandelbrot_numba_typed(xmin, xmax, ymin, ymax, width, height, max_iter=100, dtype=np.float64):
     x = np.linspace(xmin, xmax, width).astype(dtype)
     y = np.linspace(ymin, ymax, height).astype(dtype)
     results = np.zeros((height, width), dtype=np.int32)
     for i in range(height):
         for j in range(width):
             c = x[j] + 1j * y[i]
             results[i, j] = mandelbrot_point_numba(c, max_iter)
     return results
                 



if __name__ == "__main__":
    
    # Benchmarking
    """
    _ = mandelbrot_naive_numba(-2, 1, -1.5, 1.5, 1024, 1024, 100) # warmup

    t_naive = benchmark(mandelbrot_naive, -2, 1, -1.5, 1.5, 1024, 1024, 100)
    t_numpy = benchmark(compute_numpy_mandelbrot_grid, -2, 1, -1.5, 1.5, 1024, 1024, 100)
    t_numba = benchmark(mandelbrot_naive_numba, -2, 1, -1.5, 1.5, 1024, 1024, 100)

    print(f"Naive : {t_naive:.3f}s")
    print(f"NumPy : {t_numpy:.3f}s ({t_naive/t_numpy:.1f}x)")
    print(f"Numba : {t_numba:.3f}s ({t_naive/t_numba:.1f}x)")
    """
    for dtype in [np.float32, np.float64]:
         _ = mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, 100, dtype=dtype) # warmup
         t0 = time.perf_counter()
         mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, 100, dtype=dtype)
         print(f"Numba with {dtype} : {time.perf_counter() - t0:.3f}s")

    # correctness check
    """
    naive_result = compute_naive_mandelbrot_grid(-2, 1, -1.5, 1.5, 1024, 1024, 100)

    numpy_result = compute_numpy_mandelbrot_grid(-2, 1, -1.5, 1.5, 1024, 1024, 100)

    if np.allclose(naive_result, numpy_result):
        print("Results match!")
    else:
        print("Results differ!")
    """

    # Check where they differ :
    """
    diff = np.abs(naive_result - numpy_result)
    print(f"Max difference : {diff.max()}")
    print(f"Different pixels : {(diff > 0).sum()}")
    """

    #plot
    """
    plt.imshow(naive_result, cmap="hot", origin="lower")
    plt.colorbar(label="Iteration count")
    plt.title("Mandelbrot Set (naive)")
    plt.savefig("mandelbrot_naive.png")
    plt.show() 

    plt.imshow(numpy_result, cmap="hot", origin="lower")
    plt.colorbar(label="Iteration count")
    plt.title("Mandelbrot Set (NumPy)")
    plt.savefig("mandelbrot_numpy.png")
    plt.show() 
    """
    #Profiling

    """
    cProfile.run( 'compute_naive_mandelbrot_grid(-2, 1, -1.5, 1.5, 512, 512, 100)', 'naive_profile.prof' )
    cProfile.run( 'compute_numpy_mandelbrot_grid(-2, 1, -1.5, 1.5, 512, 512, 100)', 'numpy_profile.prof' )

    for name in ('naive_profile.prof', 'numpy_profile.prof'):
         stats = pstats.Stats(name)
         stats.strip_dirs()     #removes paths
         stats.sort_stats('cumulative')
         stats.print_stats(10)  
    """


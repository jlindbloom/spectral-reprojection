import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.stats import linregress




def fourier_coefficients(f, L, N=10, **kwargs):
    """Given input f(x) defined on [-L, L], returns the first N coefficients 
        k = -N/2 , ..., -N/2 + 1
    of the function f on the interval [-L,L].
    """

    assert int(N)%2 == 0, "N must be even!"
    Nhalf = int(N/2.0)

    c_k = []

    for k in range(-Nhalf, Nhalf):
        # Separate integration of real and imaginary parts
        real_part = quad(lambda x: f(x) * np.cos(k * np.pi * x / L), -L, L, **kwargs)[0]
        imag_part = quad(lambda x: f(x) * np.sin(k * np.pi * x / L), -L, L, **kwargs)[0]
        c_k_val = (1 / (2 * L)) * (real_part - 1j * imag_part)  # Combine real and imaginary parts
        c_k.append(c_k_val)

    return c_k



def fourier_partial_sum(c_k, L):
    """Returns a function representing the partial sum built using the Fourier coefficients.
    """

    N = len(c_k)
    Nhalf = int(N/2.0)

    def approximation(x):
        x = np.asarray(x)  # Ensure x is treated as a NumPy array
        sum_result = np.zeros_like(x, dtype=np.complex128)  # Initialize the sum as a complex array

        # Compute the sum over the Fourier coefficients
        for k, c in enumerate(c_k):
            # k ranges from -n_terms to n_terms, so adjust the index
            k_adjusted = k - Nhalf
            sum_result += c * np.exp(1j * k_adjusted * np.pi * x / L)

        return np.real(sum_result)  # Take the real part of the sum

    return approximation




def fourier_partial_sum_projection(f, L, N, **kwargs):
    """Given input f, interval [-L,L], and N fourier coefficients, 
    returns (S_N f, coeffs) where S_N f is the N-term partial sum approximation
    to f and coeffs are the computed fourier coefficients.
    """
    
    coeffs = fourier_coefficients(f, L, N=N, **kwargs)
    fapprox = fourier_partial_sum(coeffs, L)

    return fapprox, coeffs

















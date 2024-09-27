import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize


# Function to compute the L2 norm of the error
def L2_error(f, approx_fn, L):
    """
    Evaluates the L^2 norm error.
    """
    # Define the integrand for the error |f(x) - S_n(x)|^2
    integrand = lambda x: (f(x) - approx_fn(x))**2
    # Compute the integral and take the square root
    error, _ = quad(integrand, -L, L)
    return np.sqrt(error)


def Linf_error(f, approx_fn, L):
    """
    Evaluates the L^\infty norm error with optimization method.
    """

    def _neg_error_fn(x, f, approx_fn):
        return -np.abs(f(x) - approx_fn(x))

    # Use scipy.optimize.minimize to minimize the negative absolute error
    result = minimize(_neg_error_fn, x0=0, bounds=[(-L, L)], args=(f, approx_fn))

    # Extract the point of maximum error
    max_error_point = result.x[0]
    max_error_value = -1.0*_neg_error_fn(max_error_point, f, approx_fn)

    return max_error_value


# Function to compute the L^infty error
def Linf_error_grid(f, approx_fn, L, num_points=1000):
    """
    Evaluates the L^\infty norm error with grid method.
    """
    # Generate a grid of points in the interval [-L, L]
    x_values = np.linspace(-L, L, num_points)
    
    # Compute the absolute error at each point
    errors = np.abs(f(x_values) - approx_fn(x_values))
    
    # Return the maximum error (L^infty norm)
    return np.max(errors)











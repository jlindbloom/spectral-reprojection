import numpy as np

from scipy.integrate import quad
from scipy.special import gamma
from scipy.special import jv


def gegenbauer_polynomial(l, lambdah, x):
    """Evaluate the lth Gegenbauer polynomial at x for parameter λ."""
    if l == 0:
        return np.ones_like(x)
    elif l == 1:
        return 2 * lambdah * x
    else:
        C_lm1 = 2 * lambdah * x  # C_1(x) = 2λx
        C_lm2 = np.ones_like(x)  # C_0(x) = 1
        for n in range(2, l+1):
            C_l = ((2*(n-1+lambdah) * x * C_lm1) - ((n-2+2*lambdah) * C_lm2)) / n
            C_lm2 = C_lm1
            C_lm1 = C_l
        return C_lm1



def gegenbauer_coefficients(f, m, lambdah, **kwargs):
    """Given input f(x) defined on [-1, 1], returns the first m+1 coefficients 
        l = 0, \ldots, m
    of the function f on the interval [-1,1].
    """

    c_k = []
    weight_func = lambda x: np.power((1 - x**2), lambdah - 0.5 )

    for k in range(m+1):

        numerator = quad(lambda x: weight_func(x)*f(x)*gegenbauer_polynomial(k, lambdah, x) , -1, 1, **kwargs)[0]
        denominator = quad(lambda x: weight_func(x)*(gegenbauer_polynomial(k, lambdah, x)**2) , -1, 1, **kwargs)[0]
        c_k_val = numerator/denominator
        c_k.append(c_k_val)

    return c_k



def gegenbauer_partial_sum(gcoeffs, lambdah):
    """Returns a function representing the partial sum built using the Gegenbauer coefficients.
    """

    m = len(gcoeffs)


    def approximation(x):
        x = np.asarray(x)  # Ensure x is treated as a NumPy array
        sum_result = np.zeros_like(x, dtype=np.complex128)  # Initialize the sum as a complex array

        # Compute the sum over the Fourier coefficients
        for l, g in enumerate(gcoeffs):
            sum_result += g * gegenbauer_polynomial(l, lambdah, x)

        return np.real(sum_result)  # Take the real part of the sum

    return approximation




### From Fourier data


# Compute Gegenbauer 
def compute_gegenbauer_coeff_from_fourier(l, lambdah, fcoeffs):
    """Given the Fourier coefficients, computes the lth Gegenbauer coeff.
    """
    N = len(fcoeffs)
    Nhalf = int(N/2.0)
    f0 = fcoeffs[Nhalf+1]
    result = 0.0

    if l == 0:
        result += f0

    sum_term = 0.0
    fac = gamma(lambdah)*(np.power(1.0j, l))*(l+lambdah)
    for k in range(-Nhalf, Nhalf):
        if k == 0:
            pass
        else:
            sum_term += jv(l+lambdah, np.pi*k)*np.power( (2.0/(np.pi*k)) , lambdah )*fcoeffs[k]
    
    result += fac*sum_term

    return result




def compute_gegenbauer_coeffs_from_fourier(fcoeffs, m, lambdah):
    """Given the Fourier coefficients, computes the first m Gegenbauer coefficients.
    """
    gcoeffs = []
    for l in range(m+1):
        gcoeffs.append( compute_gegenbauer_coeff_from_fourier(l, lambdah, fcoeffs) )
    gcoeffs = np.asarray(gcoeffs)

    return gcoeffs



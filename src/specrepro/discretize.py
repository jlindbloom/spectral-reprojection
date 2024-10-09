import numpy as np

from scipy.special import gamma

from .gegenbauer import gegenbauer_polynomial



def sample_function(f, N=20):
    """Samples the function f at N points {x_j}, where
        x_j = -1 + 2(j-1)/N.
    """
    xgrid = np.asarray([-1 + 2*(j-1)/N for j in range(1,N+1)])
    return xgrid, f(xgrid)



def make_dft_matrix(N):
    """Builds the DFT matrix F.
    """

    mat = np.zeros((N,N), dtype=np.complex128)
    xjs = np.asarray([-1 + 2*(j-1)/N for j in range(1,N+1)])

    for n in range(N):
        for i in range(N):
            mat[n,i] = (1.0/N)*np.exp( -1.0j*( (n+1) - 0.5*N - 1)*np.pi*xjs[i]  )

    return mat


def make_dft_conj_transpose_matrix(N):
    """Builds the unnormalized conjugate transpose DFT matrix F.
    """

    mat = np.zeros((N,N), dtype=np.complex128)
    xjs = np.asarray([-1 + 2*(j-1)/N for j in range(1,N+1)])

    for n in range(N):
        for i in range(N):
            mat[n,i] = np.exp( 1.0j*( (n+1) - 0.5*N - 1)*np.pi*xjs[i]  )

    return mat



def make_gegenbauer_matrix(N, m, lambdah=1):

    mat = np.zeros((N,m+1))
    xjs = np.asarray([-1 + 2*(j-1)/N for j in range(1,N+1)])

    for i in range(N):
        for n in range(m+1):
            mat[i,n] = gegenbauer_polynomial( (n+1) -1, lambdah,  xjs[i])

    return mat



def gen_problem_data(f, N=30):

    xs, ys = sample_function(f, N)

    dft_mat = make_dft_matrix(N)
    dft_conj_transpose_mat = make_dft_conj_transpose_matrix(N)

    dftcoeffs = dft_mat @ ys

    return xs, ys, dftcoeffs, dft_mat, dft_conj_transpose_mat



def make_gegenbauer_matrix(N, m, lambdah=1):

    mat = np.zeros((N,m+1))
    xjs = np.asarray([-1 + 2*(j-1)/N for j in range(1,N+1)])

    for i in range(N):
        for n in range(m+1):
            mat[i,n] = gegenbauer_polynomial( (n+1) -1, lambdah,  xjs[i])

    return mat




def make_fct_matrix(N, m, lambdah=1):

    xjs = np.asarray([-1 + 2*(j-1)/N for j in range(1,N+1)])
    
    #W = np.zeros((N,N))
    #H = np.zeros((m+1,m+1))

    Wdiag = np.power( (1 - (xjs**2) ), lambdah - 0.5)
    Hdiag = np.asarray([ np.sqrt(np.pi)*gegenbauer_polynomial(l, lambdah, 1.0)*gamma(lambdah + 0.5)/(gamma(lambdah)*gamma(l+lambdah)) for l in range(1, m+2)])
    H = np.diag(Hdiag)
    W = np.diag(Wdiag)

    fgeg_mat = make_gegenbauer_matrix(N, m, lambdah=lambdah)

    fct_mat = (2.0/N)*(H @ ( fgeg_mat.T @ W ) ) 

    return fct_mat




def make_fct_and_gegenbauer_matrices(N, m, lambdah=1):

    gegenbauer_mat = make_gegenbauer_matrix(N, m, lambdah=lambdah)
    fct_mat = make_fct_matrix(N, m, lambdah=lambdah)

    return fct_mat, gegenbauer_mat



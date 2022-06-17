from numpy import * 
from scipy.special import jv, jvp, hankel1, h1vp
from scipy.linalg import block_diag, inv
# all symbols are implemented as defined in the report 

# \tilde{kappa }
def kappa_tilde(kappa, c_i, c_o, n):
        return kappa / sqrt(c_o)

# lambda^V
def lambdaV(n, kappa):
        return 1j * pi / 2.0 * jv(n, kappa) * hankel1(n, kappa)

# lambda^K
def lambdaK(n, kappa):
    return 1j * pi * kappa / 2.0 * jv(n, kappa) * h1vp(n, kappa) + 0.5

# lambda^K'
def lambdaK__adjoint(n, kappa):
    return 1j * pi * kappa / 2.0 * jv(n, kappa) * h1vp(n, kappa) + 0.5

# lambda^W
def lambdaW(n, kappa):
    return - 1j * pi * kappa ** 2 / 2.0 * jvp(n, kappa) * h1vp(n, kappa)

# c_n
def c(kappa, c_i, c_o, n):
    return (kappa_tilde(kappa, c_i, c_o, n) ** 2 + n ** 2) ** (1/2)

# method to generate block diagonal matrix from list of matrices
def blockDiagFromList(list):
    if(len(list) == 1):
        raise Exception("list must have at least 2 elements")
    if(len(list) == 2):
        return block_diag(list[0], list[1])
    return block_diag(list[0], blockDiagFromList(list[1:]))
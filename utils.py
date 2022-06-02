from numpy import * 
from scipy.special import jv, jvp, hankel1, h1vp

def kappa_tilde(kappa, c_i, c_o, n):
        return kappa / sqrt(c_o)

def lambdaV(n, kappa):
        return 1j * pi / 2.0 * jv(n, kappa) * hankel1(n, kappa)

def lambdaK(n, kappa):
    return 1j * pi * kappa / 2.0 * jv(n, kappa) * h1vp(n, kappa) + 0.5

def lambdaK__adjoint(n, kappa):
    return 1j * pi * kappa / 2.0 * jv(n, kappa) * h1vp(n, kappa) + 0.5

def lambdaW(n, kappa):
    return - 1j * pi * kappa ** 2 / 2.0 * jvp(n, kappa) * h1vp(n, kappa)
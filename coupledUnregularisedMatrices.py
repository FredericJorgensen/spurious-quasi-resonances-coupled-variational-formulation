from numpy import *
from scipy.special import jv, jvp
from eigenvaluesBIOs import lambdaV, lambdaK, lambdaK__adjoint, lambdaW

def alpha(kappa, c_i, n):
    z = kappa * sqrt(c_i)
    return jvp(n, z) / jv(n, z) * z



def getA_Tilde(kappa, c_i, n):
    a_11 = 2 * pi * (alpha(kappa, c_i, n) + lambdaW(n, kappa))
    a_12 = - 2 * pi * (0.5 - lambdaK__adjoint(n, kappa))
    a_21 = 2 * pi * (0.5 - conj(lambdaK(n, kappa)))
    a_22 = 2 * pi * (conj(lambdaV(n, kappa)))
    return array([[a_11, a_12],
                  [a_21, a_22]])




def getF_Tilde(kappa, n):
    a_11 = 0
    a_12 = - 2 * pi * (1 + lambdaW(n, kappa))
    a_21 = 2 * pi * (conj(lambdaK(n, kappa)) - 0.5)
    a_22 = 0
    return array([[a_11, a_12],
                  [a_21, a_22]])
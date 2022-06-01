from numpy import *
from scipy import *
from scipy.special import jv, jvp, hankel1, h1vp
from scipy.linalg import block_diag


class SolModel:
    def __init__(self):
        pass

    def getBlock(self, kappa, c_i, c_o, n):
        s_11 = self.lambdaK__adjoint(n, kappa) + 0.5
        s_12 = - self.lambdaV(n, kappa)
        s_21 = - self.lambdaW(n, kappa)
        s_22 = -(self.lambdaK(n, kappa) - 0.5)
        return array([[s_11, s_12],
                      [s_21, s_22]])


    def lambdaV(self, n, kappa):
        return 1j * pi / 2.0 * jv(n, kappa) * hankel1(n, kappa)

    def lambdaK(self, n, kappa):
        return 1j * pi * kappa / 2.0 * jv(n, kappa) * h1vp(n, kappa) + 0.5

    def lambdaK__adjoint(self, n, kappa):
        return 1j * pi * kappa / 2.0 * jv(n, kappa) * h1vp(n, kappa) + 0.5

    def lambdaW(self, n, kappa):
        return - 1j * pi * kappa ** 2 / 2.0 * jvp(n, kappa) * h1vp(n, kappa)

from numpy import *
from scipy.special import jv, jvp, hankel1, h1vp
from scipy.linalg import block_diag, inv
from utils import kappa_tilde, lambdaV, lambdaK, lambdaK__adjoint, lambdaW, c, blockDiagFromList
from boundary_conditions import BoundaryConditions
import matplotlib.pyplot as plt
import numpy as np

# all symbols are implemented as defined in the report

# class to generate galerkin matrix as defined in section 5


class MatrixModel:
    def __init__(self, eta=None) -> None:
        self.eta = eta

    # T_1
    def T1(self, kappa, c_i, c_o, n):
        c_n = self.c(kappa, c_i, c_o, n)
        return diag(array([1/sqrt(c_n), sqrt(c_n), 1/sqrt(c_n)]))

    # T_2
    def T2(self, kappa, c_i, c_o, n):
        c_n = self.c(kappa, c_i, c_o, n)
        return diag(array([1/sqrt(c_n), sqrt(c_n), 1/sqrt(c_n)]))

    # method that returns A_n^num as defined in section 5 OR L_eps^{-1} A_n^num as in section 7 (if input parameter removeResonances = True)
    def getBlock(self, kappa, c_i, c_o, n, removeResonances=False):

        if(self.eta == None):
            raise Exception(
                "Missing argument: The second argument is missing. There was no input provided for eta.")
        a_11 = self.alpha(kappa, c_i, c_o, n) + self.lambdaW(n, kappa)
        a_12 = -(0.5 - self.lambdaK__adjoint(n, kappa))
        a_13 = 0
        a_21 = (0.5 - self.lambdaK(n, kappa))
        a_22 = self.lambdaV(n, kappa)
        a_23 = 1j * conj(self.eta)
        a_31 = - self.lambdaW(n, kappa)
        a_32 = - (self.lambdaK__adjoint(n, kappa) + 0.5)
        a_33 = 1 + self.beta(kappa, c_i, c_o, n)

        A_B0 = array([[a_11, a_12, a_13],
                      [a_21, a_22, a_23],
                      [a_31, a_32, a_33]])
        T1 = self.T1(kappa, c_i, c_o, n)
        T2 = self.T2(kappa, c_i, c_o, n)

        if(removeResonances):
            L_eps_inv = inv(self.L_eps(kappa, c_i, c_o, n))
            return T1 @ L_eps_inv @ A_B0 @ T2
        return T1 @ A_B0 @ T2

    # L_eps
    def L_eps(self, kappa, c_i, c_o, n):
        eps = 0.01
        lambdaW = self.lambdaW(n, kappa)
        lambdaK = self.lambdaK(n, kappa)
        return array([[-1, -lambdaW, 0], [0, lambdaK - 1/2, 0], [0, lambdaW, eps]])

    # P_{V_b}
    def P_b(self, kappa, c_i, c_o, n):
        boundaryConditions = BoundaryConditions(eta=self.eta)
        return boundaryConditions.P_b(kappa, c_i, c_o, n)

    # method that returns (A_{-N}^num, ..., A_{N}^num)
    def getFullMatrix(self, kappa, c_i, c_o, N):
        A = blockDiagFromList([self.getBlock(kappa, c_i, c_o, n)
                               for n in range(-N, N + 1)])
        baseSize = 2 * N + 1
        assert(shape(A) == (3 * baseSize, 3 * baseSize))
        return A


    # v_n
    def v(self, kappa, c_i, c_o, n):
        kappa_tilde = self.kappa_tilde(kappa, c_i, c_o, n)
        return (kappa_tilde ** 2 + n ** 2) ** (-1/4)

    # w_n
    def w(self, kappa, c_i, c_o, n):
        kappa_tilde = self.kappa_tilde(kappa, c_i, c_o, n)
        return (kappa_tilde ** 2 + n ** 2) ** (1/4)

    # l_n
    def l(self, kappa, c_i, c_o, n):
        kappa_tilde = self.kappa_tilde(kappa, c_i, c_o, n)
        return(kappa_tilde ** 2 + n ** 2) ** (-1/4)

    # mathematical helper functions imported from utils.py

    def c(self, kappa, c_i, c_o, n):
        return c(kappa, c_i, c_o, n)

    def lambdaV(self, n, kappa):
        return lambdaV(n, kappa)

    def lambdaK(self, n, kappa):
        return lambdaK(n, kappa)

    def lambdaK__adjoint(self, n, kappa):
        return lambdaK__adjoint(n, kappa)

    def lambdaW(self, n, kappa):
        return lambdaW(n, kappa)
    
    def kappa_tilde(self, kappa, c_i, c_o, n):
        return kappa_tilde(kappa, c_i, c_o, n)

    #eigenvalues of bilinear forms as defined in section 5

    # alpha_n
    def alpha(self, kappa, c_i, c_o, n):
        z = kappa * sqrt(c_i/c_o)
        return z * jvp(n, z) / jv(n, z)
        
    # beta_n
    def beta(self, kappa, c_i, c_o, n):
        return n ** 2

from numpy import *
from scipy.linalg import norm
from numpy.linalg import inv
from utils import kappa_tilde, lambdaV, lambdaK, lambdaK__adjoint, lambdaW, c

# all symbols are implemented as defined in the report 

class BoundaryConditions:
    def __init__(self, eta=1):
        self.eta = eta

    def lambdaV(self, n, kappa):
        return lambdaV(n, kappa)

    def lambdaK(self, n, kappa):
        return lambdaK(n, kappa)

    def lambdaK__adjoint(self, n, kappa):
        return lambdaK__adjoint(n, kappa)

    def lambdaW(self, n, kappa):
        return lambdaW(n, kappa)
        
    def c(self, kappa, c_i, c_o, n):
        return c(kappa, c_i, c_o, n)

    def x_1(self, kappa, c_i, c_o, n):
        lambdaW = self.lambdaW(n, kappa)
        lambdaK = self.lambdaK(n, kappa)
        c_n = self.c(kappa, c_i, c_o, n)

        # x_1 = array([-(c_n)**(-0.5) * lambdaW,
        #              (c_n) ** (0.5) * (lambdaK - 0.5),
        #              (c_n)**(-0.5) * lambdaW])[newaxis].T

        # New one
        x_1 = array([-(c_n)**(-0.5) * lambdaW,
                     (c_n) ** (0.5) * (lambdaK - 0.5),
                     (c_n)**(-1) * lambdaW])[newaxis].T
        return x_1

    
    def x_2(self, kappa, c_i, c_o, n):
        c_n = self.c(kappa, c_i, c_o, n)
        x_2 = array([-(c_n) ** (-0.5), 0, 0])[newaxis].T
        return x_2

    def x_1_normed(self, kappa, c_i, c_o, n):
        x_1 = self.x_1(kappa, c_i, c_o, n)
        return x_1 / norm(x_1.flatten())

    def x_2_normed(self, kappa, c_i, c_o, n):
        x_2 = self.x_2(kappa, c_i, c_o, n)
        return x_2 / norm(x_2.flatten())

    # projector P_{V_b}
    def P_b(self, kappa, c_i, c_o, n):
        x_1_normed = self.x_1_normed(kappa, c_i, c_o, n)
        x_2_normed = self.x_2_normed(kappa, c_i, c_o, n)
        return x_1_normed @ x_1_normed.T + x_2_normed @ x_2_normed.T

    # block b_n
    def bBlock(self, kappa, c_i, c_o, f_1n, f_2n, n):
        return f_1n * self.x_1(kappa, c_i, c_o, n) + f_2n * self.x_2(kappa, c_i, c_o, n)

    # (b_-N, ..., b_N)
    def b(self, kappa, c_i, c_o, f_1n, f_2n, N):
        assert(f_1n.size == 2 * N + 1 and f_2n.size == 2 * N + 1)
        b = array([self.bBlock(kappa, c_i, c_o, f_1n[n + N], f_2n[n + N], n)
                   for n in range(-N, N+1)]).flatten()
        return b

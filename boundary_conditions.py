from matrix_model import MatrixModel
from numpy import *
from scipy.linalg import norm
from numpy.linalg import inv


class BoundaryConditions:
    def __init__(self, eta=1):
        self.model = MatrixModel(eta)
        self.eta = eta

    def x_1(self, kappa, c_i, c_o, n):
        lambdaW = self.model.lambdaW(n, kappa)
        lambdaK = self.model.lambdaK(n, kappa)
        v = self.model.v(kappa, c_i, c_o, n)
        w = self.model.w(kappa, c_i, c_o, n)
        l = self.model.l(kappa, c_i, c_o, n)
        
        x_1 = array([-lambdaW * conj(v),
                    (lambdaK - 0.5) * conj(w),  
                    lambdaW * conj(l)])[newaxis].T
        return x_1

    def x_2(self, kappa, c_i, c_o, n):
        v = self.model.v(kappa, c_i, c_o, n)
        x_2 = array([-conj(v), 0, 0])[newaxis].T
        return x_2

    def x_1_normed(self, kappa, c_i, c_o, n):
        x_1 = self.x_1(kappa, c_i, c_o, n)
        return x_1 / norm(x_1.flatten())

    def x_1_normed(self, kappa, c_i, c_o, n):
        x_2 = self.x_2(kappa, c_i, c_o, n)
        return x_2 / norm(x_2.flatten())

    def P_b(self, kappa, c_i, c_o, n):
        x_1_normed = self.x_1_normed(kappa, c_i, c_o, n)
        x_2_normed = self.x_1_normed(kappa, c_i, c_o, n)
        return x_1_normed @ x_1_normed.T + x_2_normed @ x_2_normed.T

    def bBlock(self, kappa, c_i, c_o, f_1n, f_2n, n):
        return f_1n * self.x_1(kappa, c_i, c_o, n) + f_2n * self.x_2(kappa, c_i, c_o, n)

    def b(self, kappa, c_i, c_o, f_1n, f_2n, N):
        assert(f_1n.size == 2 * N + 1 and f_2n.size == 2 * N + 1)
        b = array([self.bBlock(kappa, c_i, c_o, f_1n[n + N], f_2n[n + N], n)
                  for n in range(-N, N+1)]).flatten()
        return b

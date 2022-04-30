from matrix_model import MatrixModel
from numpy import *
from scipy.linalg import norm
from numpy.linalg import inv


class BoundaryConditions:
    def __init__(self, eta=1):
        self.model = MatrixModel("regularised", eta)
        self.eta = eta

    def x_1(self, kappa, c_i, n):
        lambdaW = self.model.lambdaW(n, kappa)
        lambdaK = self.model.lambdaK(n, kappa)
        v_tilde = self.model.v_tilde(kappa, c_i, n)
        w = self.model.w(kappa, c_i, n)
        l = self.model.l(kappa, c_i, n)
        x_1 = array([-lambdaW * conj(v_tilde),
                    (lambdaK - 0.5) * conj(w), lambdaW * conj(l)])[newaxis].T
        return x_1

    def x_2(self, kappa, c_i, n):
        v_tilde = self.model.v_tilde(kappa, c_i, n)
        x_2 = array([-conj(v_tilde), 0, 0])[newaxis].T
        return x_2

    def x_1_normed(self, kappa, c_i, n):
        x_1 = self.x_1(kappa, c_i, n)
        return x_1 / norm(x_1)

    def x_1_normed(self, kappa, c_i, n):
        x_2 = self.x_2(kappa, c_i, n)
        return x_2 / norm(x_2)

    def P_b(self, kappa, c_i, n):
        x_1_normed = self.x_1_normed(kappa, c_i, n)
        x_2_normed = self.x_1_normed(kappa, c_i, n)
        return x_1_normed @ x_1_normed.T + x_2_normed @ x_2_normed.T

    def bBlock(self, kappa, c_i, n, f_1n, f_2n):
        return f_1n * self.x_1(kappa, c_i, n) + f_2n * self.x_2(kappa, c_i, n)

    def b(self, kappa, c_i, f_1n, f_2n, N):
        assert(f_1n.size == 2 * N + 1 and f_2n.size == 2 * N + 1)
        b = array([self.bBlock(kappa, c_i, n, f_1n[n], f_2n[n])
                  for n in range(0, 2 * N + 1)]).flatten()
        return b

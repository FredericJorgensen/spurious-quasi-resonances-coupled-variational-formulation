from numpy import *
from scipy import *
from scipy.special import jv, jvp, hankel1, h1vp
from scipy.linalg import block_diag
from utils import kappa_tilde, lambdaV, lambdaK, lambdaK__adjoint, lambdaW


def blockDiagFromList(list):
    if(len(list) == 1):
        raise Exception("list must have at least 2 elements")
    if(len(list) == 2):
        return block_diag(list[0], list[1])
    return block_diag(list[0], blockDiagFromList(list[1:]))


class MatrixModel:
    def __init__(self, eta=None) -> None:
        self.eta = eta

    def getBlock(self, kappa, c_i, c_o, n):

        if(self.eta == None):
            raise Exception(
                "Missing argument: The second argument is missing. There was no input provided for eta.")
        a_11 = self.alpha(kappa, c_i, c_o, n) + self.lambdaW(n, kappa) * \
            self.P("U", "U", kappa, c_i, c_o, n)
        a_12 = -(0.5 - self.lambdaK__adjoint(n, kappa)) * \
            self.P("theta", "U", kappa, c_i, c_o, n)
        a_13 = 0
        a_21 = (0.5 - self.lambdaK(n, kappa)) * \
            self.P("U", "theta", kappa, c_i, c_o, n)
        a_22 = self.lambdaV(n, kappa) * \
            self.P("theta", "theta", kappa, c_i, c_o, n)
        a_23 = 1j * conj(self.eta) * self.P("p", "theta", kappa, c_i, c_o, n)
        a_31 = - self.lambdaW(n, kappa) * self.P("U", "p", kappa, c_i, c_o, n)
        a_32 = - (self.lambdaK__adjoint(n, kappa) + 0.5) * \
            self.P("theta", "p", kappa, c_i, c_o, n)
        a_33 = self.beta(kappa, c_i, c_o, n)
        # SIMPLIFIED VERSION FOR CONVERGENCE TESTING 
        # a_11 = self.P("U", "U", kappa, c_i, c_o, n)
        # a_12 = self.P("theta", "U", kappa, c_i, c_o, n)
        # a_13 = 0
        # a_21 = self.P("U", "theta", kappa, c_i, c_o, n)
        # a_22 = 1
        # a_23 = 0
        # a_31 = 0
        # a_32 = 0
        # a_33 = 10
        return array([[a_11, a_12, a_13],
                      [a_21, a_22, a_23],
                      [a_31, a_32, a_33]])

    def getBlockMatrix(self, kappa, c_i, c_o, N):
        A = blockDiagFromList([self.getBlock(kappa, c_i, c_o, n)
                               for n in range(-N, N + 1)])
        baseSize = 2 * N + 1
        assert(shape(A) == (3 * baseSize, 3 * baseSize))
        return A

    # mathematical helper functions
    def lambdaV(self, n, kappa):
        return lambdaV(n, kappa)

    def lambdaK(self, n, kappa):
        return lambdaK(n, kappa)

    def lambdaK__adjoint(self, n, kappa):
        return lambdaK__adjoint(n, kappa)

    def lambdaW(self, n, kappa):
        return lambdaW(n, kappa)

    def P(self, a: str, b: str, kappa: float, c_i: float, c_o, n: int):
        if(a == "U" and b == "U"):
            return 2 * pi * abs(self.v(kappa, c_i, c_o, n)) ** 2
        elif(a == "theta" and b == "U"):
            return 2 * pi * self.w(kappa, c_i, c_o, n) * conj(self.v(kappa, c_i, c_o, n))
        elif(a == "U" and b == "theta"):
            return conj(self.P("theta", "U", kappa, c_i, c_o, n))
        elif(a == "theta" and b == "theta"):
            return 2 * pi * abs(self.w(kappa, c_i, c_o, n)) ** 2
        elif(a == "p" and b == "theta"):
            return 2 * pi * self.l(kappa, c_i, c_o, n) * conj(self.w(kappa, c_i, c_o, n))
        elif(a == "U" and b == "p"):
            return 2 * pi * self.v(kappa, c_i, c_o, n) * conj(self.l(kappa, c_i, c_o, n))
        elif(a == "theta" and b == "p"):
            return conj(self.P("p", "theta", kappa, c_i, c_o, n))
        else:
            raise Exception("This scalar product is not defined.")

    def kappa_tilde(self, kappa, c_i, c_o, n):
        return kappa_tilde(kappa, c_i, c_o, n)
    
    def v(self, kappa, c_i, c_o, n):
        denominator = sqrt(2 * pi * (self.kappa_tilde(kappa, c_i, c_o, n) ** 2 + n ** 2))
        return 1 / denominator

    def w(self, kappa, c_i, c_o, n):
        return (self.kappa_tilde(kappa, c_i, c_o, n) ** 2 + n ** 2) ** (1/4) / sqrt(2 * pi)

    def l(self, kappa, c_i, c_o, n):
        denominator = sqrt(2 * pi * (self.kappa_tilde(kappa, c_i, c_o, n) ** 2 + n ** 2))
        return 1 / denominator

    # non weighted basis coefficients 
    # def v(self, kappa, c_i, c_o, n):
    #     denominator = sqrt(2 * pi * (1 + n ** 2))
    #     return 1 / denominator

    # def w(self, kappa, c_i, c_o, n):
    #     return (1 + n ** 2) ** (1/4) / sqrt(2 * pi)

    # def l(self, kappa, c_i, c_o, n):
    #     return 1/sqrt(2 * pi * (1 + n ** 2))
    def alpha(self, kappa, c_i, c_o, n):
        z = kappa * sqrt(c_i/c_o)
        # this looks weird, but it avoids overflow problem if v is very large:
        return 2 * pi * z * abs(self.v(kappa, c_i, c_o, n)) *  jvp(n, z) / jv(n, z) * abs(self.v(kappa, c_i, c_o, n))

    def beta(self, kappa, c_i, c_o, n):
        return 2 * pi * (1 + n ** 2) * abs(self.l(kappa, c_i, c_o, n)) ** 2

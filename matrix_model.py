from numpy import *
from scipy import *
from scipy.special import jv, jvp, hankel1, h1vp
from scipy.linalg import block_diag


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
        a_33 = 1
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
        return 1j * pi / 2.0 * jv(n, kappa) * hankel1(n, kappa)

    def lambdaK(self, n, kappa):
        return 1j * pi * kappa / 2.0 * jv(n, kappa) * h1vp(n, kappa) + 0.5

    def lambdaK__adjoint(self, n, kappa):
        return 1j * pi * kappa / 2.0 * jv(n, kappa) * h1vp(n, kappa) + 0.5

    def lambdaW(self, n, kappa):
        return - 1j * pi * kappa ** 2 / 2.0 * jvp(n, kappa) * h1vp(n, kappa)

    def P(self, a: str, b: str, kappa: float, c_i: float, c_o, n: int):
        if(a == "U" and b == "U"):
            return 2 * pi * abs(self.v_tilde(kappa, c_i, c_o, n)) ** 2
        elif(a == "theta" and b == "U"):
            return 2 * pi * self.w(kappa, c_i, c_o, n) * conj(self.v_tilde(kappa, c_i, c_o, n))
        elif(a == "U" and b == "theta"):
            return conj(self.P("theta", "U", kappa, c_i, c_o, n))
        elif(a == "theta" and b == "theta"):
            return 2 * pi * abs(self.w(kappa, c_i, c_o, n)) ** 2
        elif(a == "p" and b == "theta"):
            return 2 * pi * self.l(kappa, c_i, c_o, n) * conj(self.w(kappa, c_i, c_o, n))
        elif(a == "U" and b == "p"):
            return 2 * pi * self.v_tilde(kappa, c_i, c_o, n) * conj(self.l(kappa, c_i, c_o, n))
        elif(a == "theta" and b == "p"):
            return conj(self.P("p", "theta", kappa, c_i, c_o, n))
        else:
            raise Exception("This scalar product is not defined.")

    def v(self, kappa, c_i, c_o, n):
        denominator = sqrt(2 * pi * (1 + n ** 2)) * \
            abs(jv(n, kappa * sqrt(c_i/c_o)))
        return 1 / denominator

    def v_tilde(self, kappa, c_i, c_o, n):
        # avoid unnecessary 0 / 0
        # if(abs(jv(n, kappa * sqrt(c_i)) < 1e-280)):
        #    return 1 / sqrt(2 * pi * (1 + n ** 2))
        return self.v(kappa, c_i, c_o, n) * jv(n, kappa * sqrt(c_i/c_o))

    def w(self, kappa, c_i, c_o, n):
        return (1 + n ** 2) ** (1/4) / sqrt(2 * pi)

    def l(self, kappa, c_i, c_o, n):
        return 1/sqrt(2 * pi * (1 + n ** 2))

    def alpha(self, kappa, c_i, c_o, n):
        z = kappa * sqrt(c_i/c_o)
        # this looks weird, but it avoids overflow problem if v is very large:
        return 2 * pi * z * abs(self.v(kappa, c_i, c_o, n)) * jv(n, z) * abs(self.v(kappa, c_i, c_o, n)) * jvp(n, z)

    def alpha1(self, kappa, c_i, c_o, n):
        z = kappa * sqrt(c_i)
        return jvp(n, z) / jv(n, z) * z

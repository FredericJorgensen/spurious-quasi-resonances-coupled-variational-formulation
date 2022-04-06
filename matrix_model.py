from numpy import *
from scipy import *
from scipy.special import jv, jvp
from scipy.special import jv, jvp, hankel1, h1vp


class MatrixModel:
    def __init__(self, model="unregularised", eta=None) -> None:
        # model has to be "regularised" || "unregularised"
        self.model = model
        self.eta = eta

    def getA_Tilde(self, kappa, c_i, n):
        if(self.model == "unregularised"):
            a_11 = 2 * pi * (self.alpha(kappa, c_i, n) + self.lambdaW(n, kappa))
            a_12 = - 2 * pi * (0.5 - self.lambdaK__adjoint(n, kappa))
            a_21 = 2 * pi * (0.5 - conj(self.lambdaK(n, kappa)))
            a_22 = 2 * pi * (conj(self.lambdaV(n, kappa)))
            return array([[a_11, a_12],
                          [a_21, a_22]])
        elif(self.model == "regularised"):
            if(self.eta == None):
                raise Exception(
                    "Missing argument: The second argument is missing. There was no input provided for eta.")
            a_11 = 2 * pi *self.R__der__at__1(kappa, c_i, n) * self.R__at__1(kappa,
                                                                    c_i, n) + self.lambdaW(n, kappa) * self.P("V", "V", kappa, c_i, n)
            a_12 = (0.5 - self.lambdaK__adjoint(n, kappa)) * \
                self.P("y", "V", kappa, c_i, n)
            a_13 = 0
            a_21 = conj(self.lambdaK__adjoint(n, kappa) - 0.5) * \
                self.P("y", "V", kappa, c_i, n)
            a_22 = conj(self.lambdaV(n, kappa) * self.P("y", "y", kappa, c_i, n))
            a_23 = conj(1j * self.eta * self.P("y", "y", kappa, c_i, n))
            a_31 = self.lambdaW(n, kappa) * self.P("V", "y", kappa, c_i, n)
            a_32 = - (self.lambdaK__adjoint(n, kappa) + 0.5) * \
                self.P("y", "y", kappa, c_i, n)
            a_33 = 2 * pi * (1 + n ** 2)
            return array([[a_11, a_12, a_13],
                          [a_21, a_22, a_23],
                          [a_31, a_32, a_33]])
        else:
            raise Exception("Type Error: the given model does not exist.")

    # mathematical helper functions
    def lambdaV(self, n, kappa):
        return 1j * pi / 2.0 * jv(n, kappa) * hankel1(n, kappa)

    def lambdaK(self, n, kappa):
        return 1j * pi * kappa / 2.0 * jv(n, kappa) * h1vp(n, kappa) + 0.5

    def lambdaK__adjoint(self, n, kappa):
        return 1j * pi * kappa / 2.0 * jv(n, kappa) * h1vp(n, kappa) + 0.5

    def lambdaW(self, n, kappa):
        return - 1j * pi * kappa ** 2 / 2.0 * jvp(n, kappa) * h1vp(n, kappa)

    def R__at__1(self, kappa, c_i, n) -> float:
        return 1

    def R__der__at__1(self, kappa, c_i, n) -> float:
        z = kappa ** 2 * c_i
        factor1 = self.R__at__1(kappa, c_i, n)
        factor2 = jvp(n, z) / jv(n, z) * z
        return factor1 * factor2

    def P(self, a: str, b: str, kappa: float, c_i: float, n: int) -> float:
        if(a == "V" and b == "V"):
            return 2 * pi * self.R__at__1(kappa, c_i, n) ** 2
        elif(a == "y" and b == "V"):
            return 2 * pi * self.R__at__1(kappa, c_i, n)
        elif(a == "V" and b == "y"):
            return 2 * pi * conj(self.R__at__1(kappa, c_i, n))
        elif(a == "y" and b == "y"):
            return 2 * pi
        else:
            raise Exception("This scalar product is not defined.")

    def alpha(self, kappa, c_i, n):
        z = kappa * sqrt(c_i)
        return jvp(n, z) / jv(n, z) * z

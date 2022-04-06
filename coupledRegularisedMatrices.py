from numpy import *
from scipy.special import jv, jvp
from eigenvaluesBIOs import lambdaV, lambdaK, lambdaK__adjoint, lambdaW


def R__at__1(kappa, c_i, n) -> float:
    return 1


def R__der__at__1(kappa, c_i, n) -> float:
    z = kappa ** 2 * c_i
    factor1 = R__at__1(kappa, c_i, n)
    factor2 = jvp(n, z) / jv(n, z) * z
    return factor1 * factor2


def P(a: str, b: str, kappa: float, c_i: float, n: int) -> float:
    if(a == "V" and b == "V"):
        return 2 * pi * R__at__1(kappa, c_i, n) ** 2
    elif(a == "y" and b == "V"):
        return 2 * pi * R__at__1(kappa, c_i, n)
    elif(a == "V" and b == "y"):
        return 2 * pi * conj(R__at__1(kappa, c_i, n))
    elif(a == "y" and b == "y"):
        return 2 * pi
    else:
        raise Exception("This scalar product is not defined.")


def getA_Tilde(kappa, c_i, n, eta=1.0):
    a_11 = 2 * pi * R__der__at__1(kappa, c_i, n) * R__at__1(kappa,
                                                            c_i, n) + lambdaW(n, kappa) * P("V", "V", kappa, c_i, n)
    a_12 = (0.5 - lambdaK__adjoint(n, kappa)) * P("y", "V", kappa, c_i, n)
    a_13 = 0
    a_21 = conj(lambdaK__adjoint(n, kappa) - 0.5) * \
        P("y", "V", kappa, c_i, n)
    a_22 = conj(lambdaV(n, kappa) * P("y", "y", kappa, c_i, n))
    a_23 = conj(1j * eta * P("y", "y", kappa, c_i, n))
    a_31 = lambdaW(n, kappa) * P("V", "y", kappa, c_i, n)
    a_32 = - (lambdaK__adjoint(n, kappa) + 0.5) * \
        P("y", "y", kappa, c_i, n)
    a_33 = 2 * pi * (1 + n ** 2)
    return array([[a_11, a_12, a_13],
                  [a_21, a_22, a_23],
                  [a_31, a_32, a_33]])

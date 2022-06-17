from numpy import *
from scipy import *
from scipy.special import jv, jvp, hankel1, h1vp
from scipy.linalg import block_diag
from utils import kappa_tilde, lambdaV, lambdaK, lambdaK__adjoint, lambdaW

# class that implements analytical solution operator of the considered Helmholtz transmission problem
class SolModel:
    def __init__(self):
        pass
    
    # return nth block of solution operator 
    def getBlock(self, kappa, c_i, c_o, n):
        kappa_tilde = self.kappa_tilde(kappa, c_i, c_o, n)
        ki = kappa_tilde * sqrt(c_i)
        ko = kappa_tilde * sqrt(c_o)

        s_11 = - jv(n, ki) * ko * h1vp(n, ko)
        s_12 = jv(n, ki) * hankel1(n, ko) * sqrt(n ** 2 + kappa_tilde ** 2)
        s_21 = - ki * jvp(n, ki) * ko * h1vp(n, ko) / sqrt(n ** 2 + kappa_tilde ** 2)
        s_22 = ki * jvp(n, ki) * hankel1(n, ko)
        factor = 1 / (hankel1(n, ko) * ki * jvp(n, ki) - jv(n, ki) * ko * h1vp(n, ko))
        return factor * array([[s_11, s_12],
                              [s_21, s_22]])

    def kappa_tilde(self, kappa, c_i, c_o, n):
        return kappa_tilde(kappa, c_i, c_o, n)

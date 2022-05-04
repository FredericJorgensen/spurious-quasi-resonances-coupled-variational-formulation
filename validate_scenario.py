#!/usr/bin/env python3
from numpy import *
from scipy.special import jv, jvp, hankel1, h1vp

kappa = 1

H0 = hankel1(0, kappa)
H1 = hankel1(1, kappa)

J0 = jv(0, kappa)
J1 = jv(1, kappa)


lambdaW = 1j * pi / 2 * J1 * H1
lambdaK = - 1j * pi /2 * J0 * H1 + 0.5


lhs = - J1 + lambdaW * J0 + (0.5 - lambdaK) * H1

rhs = -J1 + H1 - lambdaW * (H0 - J0)



print("lhs: ", lhs)
print("rhs: ", rhs)
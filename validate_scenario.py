#!/usr/bin/env python3
from numpy import *
from scipy.special import jv, jvp, hankel1, h1vp

kappa = 5.502

H0 = hankel1(0, kappa)
H1 = hankel1(1, kappa)

J0 = jv(0, kappa)
J1 = jv(1, kappa)

J0prime = jvp(0, kappa)
H0prime = h1vp(0, kappa)

lambdaW = - 1j * pi * kappa ** 2 / 2 * J0prime * H0prime
lambdaK = 1j * pi * kappa /2 * J0 * H0prime + 0.5
lambdaV = 1j * pi / 2.0 * J0 * H0


lhs1 = - kappa * J1 + lambdaW * J0 + kappa * (0.5 - lambdaK) * H1
rhs1 = - kappa * (J1 - H1) - lambdaW * (H0 - J0)

lhs2 = (0.5 - lambdaK) * J0 + lambdaV * kappa * H0prime
rhs2 = (lambdaK - 0.5 ) * (H0 - J0)

lhs3 = - lambdaW * J0 - (lambdaK + 0.5) * kappa * H0prime
rhs3 = lambdaW * (H0 - J0)


print("lhs1: ", lhs1)
print("rhs1: ", rhs1)
print("-------------------")
print("lhs2: ", lhs2)
print("rhs2: ", rhs2)
print("-------------------")
print("lhs3: ", lhs3)
print("rhs3: ", rhs3)
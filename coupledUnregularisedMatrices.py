#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 16:58:43 2022

@author: fredericjoergensen
"""

from numpy import *
from scipy.special import jv, jvp
from eigenvaluesBIOs import lambdaV, lambdaK, lambdaK__adjoint, lambdaW

def alpha(kappa, c_i, n):
    z = kappa * sqrt(c_i)
    return jvp(n, z) / jv(n, z) * z



def getATilde(kappa, c_i, n):
    a_11 = 2 * pi * (alpha(kappa, c_i, n) + lambdaW(n, kappa))
    a_12 = - 2 * pi * (0.5 - lambdaK__adjoint(n, kappa))
    a_21 = 2 * pi * (0.5 - conj(lambdaK(n, kappa)))
    a_22 = 2 * pi * (conj(lambdaV(n, kappa)))
    return 2 * pi * array([[a_11, a_12],
                           [a_21, a_22]])



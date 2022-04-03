#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 16:58:04 2022

@author: fredericjoergensen
"""
from numpy import *
from scipy import *
from scipy.special import jv, jvp, hankel1, h1vp


def lambdaV(n, kappa):
    return 1j * pi / 2.0 * jv(n, kappa) * hankel1(n, kappa)

def lambdaK(n, kappa):
    return 1j * pi * kappa / 2.0 * jv(n, kappa) * h1vp(n, kappa) + 0.5

def lambdaK__adjoint(n, kappa):
    return lambdaK(n, kappa)

def lambdaW(n, kappa):
    return  - 1j * pi * kappa ** 2 / 2.0 * jvp(n, kappa) * h1vp(n, kappa) #
#why flipped sign???

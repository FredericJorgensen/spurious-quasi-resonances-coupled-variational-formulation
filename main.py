#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 16:52:26 2022

@author: fredericjoergensen
"""
from coupledUnregularisedMatrices import getA_Tilde
from numpy import *
import matplotlib.pyplot as plt


def getMaximumSingularValue(kappa, c_i, N):
    maxSigma = 0
    for n in range(-N, N + 1):
        A_Tilde= getA_Tilde(kappa, c_i, n)
        u, s, vh = linalg.svd(A_Tilde)
        maxSigma = max(maxSigma, s.max())
    return maxSigma

def getMinimumSingularValue(kappa, c_i, N):
    minSigma = float(inf)
    for n in range(-N, N + 1):
        A_Tilde= getA_Tilde(kappa, c_i, n)
        u, s, vh = linalg.svd(A_Tilde)
        minSigma = min(minSigma, s.min())
    return minSigma


def plot(x, y, xLabelName, yLabelName):
    plt.figure()
    plt.plot(x,y)
    plt.xlabel(xLabelName)
    plt.ylabel(yLabelName)
    plt.show()



if __name__ == "__main__":
    c_i = 1.3
    N = 100
    kVals = linspace(2.0, 10.0, 100)
    operatorNorms = zeros_like(kVals)
    for (index, kappa) in enumerate(kVals):
        operatorNorms[index] = getMinimumSingularValue(kappa, c_i, N)
    plot(kVals, operatorNorms, r"$\kappa$", r"$\sigma_{max}$")
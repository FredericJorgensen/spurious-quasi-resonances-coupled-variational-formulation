#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 16:52:26 2022

@author: fredericjoergensen
"""
from coupledUnregularisedMatrices import getATilde
from numpy import *
import matplotlib.pyplot as plt


def getMaximumSingularValue(kappa, c_i, N):
    maxSigma = 0
    for n in range(-N, N + 1):
        aTilde= getATilde(kappa, c_i, n)
        u, s, vh = linalg.svd(aTilde)
        maxSigma = max(maxSigma, s.max())
    return maxSigma

def getMinimumSingularValue(kappa, c_i, N):
    minSigma = float(inf)
    for n in range(-N, N + 1):
        aTilde= getATilde(kappa, c_i, n)
        u, s, vh = linalg.svd(aTilde)
        minSigma = min(minSigma, s.min())
    return minSigma


def plot(x, y):
    plt.figure()
    plt.plot(x,y)
    plt.show()



if __name__ == "__main__":
    c_i = 1.3
    N = 100
    kVals = linspace(2.0, 10.0, 100)
    operatorNorms = zeros_like(kVals)
    for (index, kappa) in enumerate(kVals):
        operatorNorms[index] = getMinimumSingularValue(kappa, c_i, N)
    plot(kVals, operatorNorms)
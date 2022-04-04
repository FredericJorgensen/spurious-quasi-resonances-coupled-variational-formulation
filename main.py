#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 16:52:26 2022

@author: fredericjoergensen
"""
from coupledUnregularisedMatrices import getA_Tilde, getF_Tilde
from numpy import *
import matplotlib.pyplot as plt


def getSingularValueOfBlock(kappa, c_i, n, index = None):
    #returns singular values of A_Tilde_b
    A_Tilde= getA_Tilde(kappa, c_i, n)
    u, s, vh = linalg.svd(A_Tilde)
    if(index != None):
        return s[index]
    return s


def getMaximumSingularValue(kappa, c_i, N):
    #returns maximum singular value of matrices A_Tilde_n with n in [-N, N]
    maxSigma = 0
    for n in range(-N, N + 1):
        s = getSingularValueOfBlock(kappa, c_i, n)
        maxSigma = max(maxSigma, s.max())
    return maxSigma

def getMinimumSingularValue(kappa, c_i, N):
    #returns minimum singular value of matrices A_Tilde_n with n in [-N, N]
    minSigma = float(inf)
    for n in range(-N, N + 1):
        s = getSingularValueOfBlock(kappa, c_i, n)
        minSigma = min(minSigma, s.min())
    return minSigma



def plot(x, y, xLabelName, yLabelName, plotName):
    plt.figure()
    plt.plot(x,y)
    plt.xlabel(xLabelName)
    plt.ylabel(yLabelName)
    plt.savefig("./figures/" + plotName + ".pdf")



def simulate(scenarioName, scenarioMethod, c_i,  n = None, N = None, index = None):
    plotName = scenarioName + "c_i" + str(c_i)
    if(n):
        plotName += "n_" + str(n)
    if(N):
        plotName += "N_" + str(N)
    if(index):
        plotName += "index_" + str(index)

    kappaVals = linspace(2.0, 10.0, 100)
    sVals = zeros_like(kappaVals)

    for (i, kappa) in enumerate(kappaVals):
        if(N):
            sVals[i] = scenarioMethod(kappa, c_i, N)
        elif(n):
            sVals[i] = scenarioMethod(kappa, c_i, n, index)

    plot(kappaVals, sVals, r"$\kappa$", r"$\sigma_{max}$", plotName)

if __name__ == "__main__":

    simulate("getMaximumSingularValue", getMaximumSingularValue, 3.0, N = 100)

    simulate("getMinimumSingularValue", getMinimumSingularValue, 3.0, N = 100)

    simulate("getSingularValueOfBlock", getSingularValueOfBlock, 3.0, n = 5, index = 0)
    simulate("getSingularValueOfBlock", getSingularValueOfBlock, 3.0, n = 5, index = 1)

    simulate("getSingularValueOfBlock", getSingularValueOfBlock, 1.0, n = 5, index = 0)
    simulate("getSingularValueOfBlock", getSingularValueOfBlock, 1.0, n = 5, index = 1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Apr  3 16:52:26 2022

@author: fredericjoergensen
"""
from coupledUnregularisedMatrices import getA_Tilde, getF_Tilde
from numpy import *
import matplotlib.pyplot as plt
from scipy.special import jn_zeros


def getSingularValueOfBlock(kappa, c_i, n, index = None):
    #returns singular values of A_Tilde_b
    A_Tilde= getA_Tilde(kappa, c_i, n)
    u, s, vh = linalg.svd(A_Tilde)
    if(index != None):
        return s[index]
    return s


def getMaximumSingularValue(kappa, c_i, N):
    #

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

def ratioMaximumMinimumSingularValue(kappa, c_i, N):
    return getMaximumSingularValue(kappa, c_i, N) / getMinimumSingularValue(kappa, c_i, N)


def getBesselRootsFromInterval(a, b, N):
    roots = array([])
    for i in range(0,5):
        rootsTemp = array([])
        y = a
        rootsTemp = jn_zeros(i, 10)
        roots = concatenate((roots, rootsTemp))
    return roots[logical_and(roots < b, roots > a)]


def plot(x, y, xLabelName, yLabelName, plotName):
    plt.figure()
    plt.plot(x,y, 'r')
    plt.xlabel(xLabelName)
    plt.ylabel(yLabelName)


def getPlotName(scenarioName, c_i, n = None, N = None, index = None,
                plotBesselRoots = False, plotRange = [2.0, 10.0]):
    plotName = scenarioName + "c_i" + str(c_i)
    if(n):
        plotName += "n_" + str(n)
    if(N):
        plotName += "N_" + str(N)
    if(index):
        plotName += "index_" + str(index)
    if(plotBesselRoots):
        plotName += "plotBesselRoots_" + "true"
    if(plotRange[0] != 2.0 or plotRange[1] != 10.0):
        plotName += "plotRangeStart_" + str(plotRange[0]) + "plotRangeEnd_" + str(plotRange[1])
    return plotName



def simulate(scenarioMethod, c_i,  n = None, N = None,
                 index = None, plotRange = [2.0, 10.0]):
    kappaVals = linspace(plotRange[0], plotRange[1], 100)
    sVals = zeros_like(kappaVals)

    for (i, kappa) in enumerate(kappaVals):
        if(N):
            sVals[i] = scenarioMethod(kappa, c_i, N)
        elif(n):
            sVals[i] = scenarioMethod(kappa, c_i, n, index)
    return kappaVals, sVals


def plotScenario(scenarioName, scenarioMethod, c_i,  n = None, N = None,
                 index = None, plotBesselRoots = False, plotRange = [2.0, 10.0],
                 yLabelName = r"$\sigma$"):

    plotName = getPlotName(scenarioName, c_i, n, N, index, plotBesselRoots, plotRange)

    kappaVals, sVals = simulate(scenarioMethod, c_i, n, N, index, plotRange)

    plot(kappaVals, sVals, r"$\kappa$", yLabelName, plotName)
    if(plotBesselRoots):
        if(N == None):
            raise Exception("N needs to be defined to plot bessel roots")
        rescalePlotRange = array(plotRange) * sqrt(c_i)
        besselRoots = getBesselRootsFromInterval(rescalePlotRange[0], rescalePlotRange[1], N)
        maxVal = sVals.max()
        plt.vlines(besselRoots / sqrt(c_i), zeros_like(besselRoots),
                   maxVal * ones_like(besselRoots), linestyles='dashed')
    plt.savefig("./figures/" + plotName + ".pdf")







if __name__ == "__main__":

    plotScenario("ratioMaximumMinimumSingularValue", ratioMaximumMinimumSingularValue,
                 3.0, N = 200, plotRange = [6.80, 6.82], plotBesselRoots= True,
                 yLabelName = r"$\sigma_{max} / \sigma_{min}$")
    #plotScenario("ratioMaximumMinimumSingularValue", ratioMaximumMinimumSingularValue,
                 #3.0, N = 200, plotRange = [6.0, 8.0], plotBesselRoots= True,
                 #yLabelName = r"$\sigma_{max} / \sigma_{min}$")

    #plotScenario("getMaximumSingularValue", getMaximumSingularValue, 3.0, N = 100)

    #plotScenario("getMinimumSingularValue", getMinimumSingularValue, 3.0, N = 100)

    #plotScenario("getSingularValueOfBlock", getSingularValueOfBlock, 3.0, n = 5, index = 0)
    #plotScenario("getSingularValueOfBlock", getSingularValueOfBlock, 3.0, n = 5, index = 1)

    #plotScenario("getSingularValueOfBlock", getSingularValueOfBlock, 1.0, n = 5, index = 0)
    #plotScenario("getSingularValueOfBlock", getSingularValueOfBlock, 1.0, n = 5, index = 1)



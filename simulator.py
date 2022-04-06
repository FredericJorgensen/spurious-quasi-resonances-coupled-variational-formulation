from matrix_model import MatrixModel
from numpy import *
import matplotlib.pyplot as plt
from scipy.special import jn_zeros


class Simulator:
    def __init__(self, model, eta=None) -> None:
        self.model = MatrixModel(model, eta)

    # define scenarios

    def getSingularValueOfBlock(self, kappa, c_i, n, index=None):
        # returns singular values of A_Tilde_b
        A_Tilde = self.model.getA_Tilde(kappa, c_i, n)
        u, s, vh = linalg.svd(A_Tilde)
        if(index != None):
            return s[index]
        return s

    def ratioMaximumMinimumSingularValue(self, kappa, c_i, N):
        return self.getMaximumSingularValue(kappa, c_i, N) / self.getMinimumSingularValue(kappa, c_i, N)

    def getMaximumSingularValue(self, kappa, c_i, N):
        #

        maxSigma = 0
        for n in range(-N, N + 1):
            s = self.getSingularValueOfBlock(kappa, c_i, n)
            maxSigma = max(maxSigma, s.max())
        return maxSigma

    def getMinimumSingularValue(self, kappa, c_i, N):
        # returns minimum singular value of matrices A_Tilde_n with n in [-N, N]
        minSigma = float(inf)
        for n in range(-N, N + 1):
            s = self.getSingularValueOfBlock(kappa, c_i, n)
            minSigma = min(minSigma, s.min())
        return minSigma

    # bessel roots
    def getBesselRootsFromInterval(self, a, b, N):
        roots = array([])
        for i in range(0, 5):
            rootsTemp = array([])
            y = a
            rootsTemp = jn_zeros(i, 10)
            roots = concatenate((roots, rootsTemp))
        return roots[logical_and(roots < b, roots > a)]

    # plotting of scenarios

    def simulate(self, scenarioMethod, c_i,  n=None, N=None,
                 index=None, plotRange=[2.0, 10.0]):
        kappaVals = linspace(plotRange[0], plotRange[1], 100)
        sVals = zeros_like(kappaVals)

        for (i, kappa) in enumerate(kappaVals):
            if(N):
                sVals[i] = scenarioMethod(kappa, c_i, N)
            elif(n):
                sVals[i] = scenarioMethod(kappa, c_i, n, index)
        return kappaVals, sVals

    def plotScenario(self, scenarioName, c_i,  n=None, N=None, plotRange=[2.0, 10.0],
                     index=None, plotBesselRoots=False):

        scenarioMethodOf = {"getSingularValueOfBlock": self.getSingularValueOfBlock,
                            "ratioMaximumMinimumSingularValue": self.ratioMaximumMinimumSingularValue,
                            "getMinimumSingularValue": self.getMinimumSingularValue,
                            "getMaximumSingularValue": self.getMaximumSingularValue}

        yLabelNameOf = {"getSingularValueOfBlock": r"$\sigma$",
                        "ratioMaximumMinimumSingularValue": r"$\sigma_{max} / \sigma_{min}$",
                        "getMinimumSingularValue": r"$\sigma_{min}$",
                        "getMaximumSingularValue": r"$\sigma_{max}$"}

        scenarioMethod = scenarioMethodOf[scenarioName]
        yLabelName = yLabelNameOf[scenarioName]

        plotName = self.getPlotName(scenarioName, c_i, n, N,
                                    index, plotBesselRoots, plotRange)

        kappaVals, sVals = self.simulate(
            scenarioMethod, c_i, n, N, index, plotRange)

        self.plot(kappaVals, sVals, r"$\kappa$", yLabelName)

        if(plotBesselRoots):
            if(N == None):
                raise Exception("N needs to be defined to plot bessel roots")
            rescalePlotRange = array(plotRange) * sqrt(c_i)
            besselRoots = self.getBesselRootsFromInterval(
                rescalePlotRange[0], rescalePlotRange[1], N)
            maxVal = sVals.max()
            plt.vlines(besselRoots / sqrt(c_i), zeros_like(besselRoots),
                       maxVal * ones_like(besselRoots), linestyles='dashed')
        plt.savefig("./figures/" + plotName + ".pdf")

    # methods for plotting

    def plot(self, x, y, xLabelName, yLabelName):
        plt.figure()
        plt.plot(x, y, 'r')
        plt.xlabel(xLabelName)
        plt.ylabel(yLabelName)

    def getPlotName(self, scenarioName, c_i, n, N, index,
                    plotBesselRoots, plotRange):
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
            plotName += "plotRangeStart_" + \
                str(plotRange[0]) + "plotRangeEnd_" + str(plotRange[1])
        return plotName

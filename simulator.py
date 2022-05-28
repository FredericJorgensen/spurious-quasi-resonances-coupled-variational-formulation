from matrix_model import MatrixModel
from numpy import *
import matplotlib.pyplot as plt
from scipy.special import jn_zeros, jv, jvp, hankel1, h1vp
from sol_model import SolModel


class Simulator:
    def __init__(self, model= "variational", eta=None) -> None:
        if(model == "variational"):
            self.model = MatrixModel(eta)
        elif(model == "solution"):
            self.model = SolModel()
        else:
            raise Exception("Invalid model")
        self.modelName = model
        self.eta = eta

    # define scenarios

    def getSingularValueOfBlock(self, kappa, c_i, c_o, n, index=None):
        # returns singular values of A_Tilde_b
        A_Tilde = self.model.getBlock(kappa, c_i, c_o, n)
        u, s, vh = linalg.svd(A_Tilde)
        if(index != None):
            return s[index]
        return s

    def ratioMaximumMinimumSingularValue(self, kappa, c_i, c_o, N):
        return self.getMaximumSingularValue(kappa, c_i, c_o, N) / self.getMinimumSingularValue(kappa, c_i, c_o, N)

    def getMaximumSingularValue(self, kappa, c_i, c_o, N):
        maxSigma = 0
        for n in range(-N, N + 1):
            s = self.getSingularValueOfBlock(kappa, c_i, c_o, n)
            maxSigma = max(maxSigma, s.max())
        return maxSigma

    def getMinimumSingularValue(self, kappa, c_i, c_o, N):
        # returns minimum singular value of matrices A_Tilde_n with n in [-N, N]
        minSigma = float(inf)
        for n in range(-N, N + 1):
            s = self.getSingularValueOfBlock(kappa, c_i, c_o, n)
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

    def simulate(self, scenarioMethod, c_i, c_o,  n=None, N=None,
                 index=None, plotRange=[2.0, 10.0]):
        kappaVals = linspace(plotRange[0], plotRange[1], 100)
        sVals = zeros_like(kappaVals)

        for (i, kappa) in enumerate(kappaVals):
            if(N):
                sVals[i] = scenarioMethod(kappa, c_i, c_o, N)
            else:
                raise Exception("Invalid arguments")
            # elif(n):
                # sVals[i] = scenarioMethod(kappa, c_i, c_o, n, index)
        return kappaVals, sVals

    def plotScenario(self, scenarioName, c_i, c_o,  n=None, N=None, plotRange=[2.0, 10.0],
                     index=None, plotBesselRoots=False):

        scenarioMethodOf = {"getSingularValueOfBlock": self.getSingularValueOfBlock,
                            "ratioMaximumMinimumSingularValue": self.ratioMaximumMinimumSingularValue,
                            "MinimumSingularValue": self.getMinimumSingularValue,
                            "MaximumSingularValue": self.getMaximumSingularValue}

        yLabelNameOf = {"getSingularValueOfBlock": r"$\sigma$",
                        "ratioMaximumMinimumSingularValue": r"$\sigma_{max} / \sigma_{min}$",
                        "MinimumSingularValue": r"$\sigma_{min}$",
                        "MaximumSingularValue": r"$\sigma_{max}$"}

        scenarioMethod = scenarioMethodOf[scenarioName]
        yLabelName = yLabelNameOf[scenarioName]

        plotName = self.getPlotName(scenarioName, c_i, c_o, n, N,
                                    index, plotBesselRoots, plotRange)

        kappaVals, sVals = self.simulate(
            scenarioMethod, c_i, c_o, n, N, index, plotRange)

        self.plot(kappaVals, sVals, r"$\kappa$", yLabelName)

        if(plotBesselRoots):
            if(N == None):
                raise Exception("N needs to be defined to plot bessel roots")
            rescalePlotRange = array(plotRange) * sqrt(c_i) / sqrt(c_o)
            besselRoots = self.getBesselRootsFromInterval(
                rescalePlotRange[0], rescalePlotRange[1], N)
            maxVal = sVals.max()
            plt.vlines(besselRoots * sqrt(c_o) / sqrt(c_i), zeros_like(besselRoots),
                       maxVal * ones_like(besselRoots), linestyles='dashed')
        plt.savefig("./figures/" + self.modelName + "/" + plotName + ".pdf")

    # methods for plotting

    def plot(self, x, y, xLabelName, yLabelName):
        plt.figure()
        plt.semilogy(x, y, 'r')
        plt.xlabel(xLabelName)
        plt.ylabel(yLabelName)
        plt.grid(True)

    def getPlotName(self, scenarioName, c_i, c_o, n, N, index,
                    plotBesselRoots, plotRange):
        plotName = scenarioName + "c_i" + str(c_i) + "c_o" + str(c_o)
        if(n):
            plotName += "n_" + str(n)
        if(N):
            plotName += "N_" + str(N)
        if(index):
            plotName += "index_" + str(index)
        if(plotBesselRoots):
            plotName += "plotBesselRoots_" + "true"
        if(self.eta):
            plotName += "eta" + str(self.eta)
        if(plotRange[0] != 2.0 or plotRange[1] != 10.0):
            plotName += "plotRangeStart_" + \
                str(plotRange[0]) + "plotRangeEnd_" + str(plotRange[1])
        return plotName

# %%

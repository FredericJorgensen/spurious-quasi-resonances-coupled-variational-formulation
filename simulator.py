from matrix_model import MatrixModel
from numpy import *
import matplotlib.pyplot as plt
from scipy.special import jn_zeros, jv, jvp, hankel1, h1vp, jnjnp_zeros, jnp_zeros
from sol_model import SolModel

numberOfValues = 1000


class Simulator:
    def __init__(self, eta=None) -> None:

        self.matrixModel = MatrixModel(eta)
        self.solutionModel = SolModel()
        self.modelName = "variational"
        self.eta = eta

    # define scenarios

    def getEuclidianNormOfBlock(self, kappa, c_i, c_o, n, removeResonances=False):
        matrixBlock = self.solutionModel.getBlock(kappa, c_i, c_o, n)
        return linalg.norm(matrixBlock)

    def getSingularValueOfBlock(self, kappa, c_i, c_o, n, index=None, removeResonances=False):
        # returns singular values of matrixBlock
        matrixBlock = self.matrixModel.getBlock(
            kappa, c_i, c_o, n, removeResonances=removeResonances)
        u, s, vh = linalg.svd(matrixBlock)
        if(index != None):
            return s[index]
        print(matrixBlock)
        return s

    def ratioMaximumMinimumSingularValue(self, kappa, c_i, c_o, N):
        return self.getMaximumSingularValue(kappa, c_i, c_o, N) / self.getMinimumSingularValue(kappa, c_i, c_o, N), 0

    def getMaximumEuclidianNorm(self, kappa, c_i, c_o, N, removeResonances=False):
        maxNorm = 0
        maxIndex = -N
        for n in range(-N, N + 1):
            newNorm = self.getEuclidianNormOfBlock(kappa, c_i, c_o, n)
            if(maxNorm < newNorm):
                maxIndex = n
            maxNorm = max(maxNorm, newNorm)
        return maxNorm, maxIndex

    def getMaximumSingularValue(self, kappa, c_i, c_o, N):
        maxSigma = 0
        maxIndex = -N
        for n in range(-N, N + 1):
            s = self.getSingularValueOfBlock(kappa, c_i, c_o, n)
            if(maxSigma < s.max()):
                maxIndex = n
            maxSigma = max(maxSigma, s.max())
        return maxSigma, maxIndex

    def getMinimumSingularValue(self, kappa, c_i, c_o, N):
        # returns minimum singular value of matrices matrixBlock_n with n in [-N, N]
        minSigma = float(inf)
        minIndex = -N
        for n in range(-N, N + 1):
            s = self.getSingularValueOfBlock(kappa, c_i, c_o, n)
            if(minSigma > s.min()):
                minIndex = n
            minSigma = min(minSigma, s.min())
        return minSigma, minIndex

    def getInvertedMinimumSingularValue(self, kappa, c_i, c_o, N, removeResonances=False):
        # returns minimum singular value of matrices matrixBlock_n with n in [-N, N]
        minSigma = float(inf)
        minIndex = -N
        for n in range(-N, N + 1):
            s = self.getSingularValueOfBlock(
                kappa, c_i, c_o, n, removeResonances=removeResonances)
            if(minSigma > s.min()):
                minIndex = n
            minSigma = min(minSigma, s.min())
        return 1 / minSigma, minIndex

    # bessel roots
    def getBesselRootsFromInterval(self, a, b, N):
        roots = array([])
        for i in range(-10, 10):
            rootsTemp = array([])
            rootsTemp = jn_zeros(i, 10)
            roots = concatenate((roots, rootsTemp))
        return roots[logical_and(roots < b, roots > a)]

    # plotting of scenarios

    def simulate(self, scenarioMethod, c_i, c_o,  n=None, N=None,
                 index=None, plotRange=[2.0, 10.0], removeResonances=False):
        kappaTildeVals = linspace(plotRange[0], plotRange[1], numberOfValues)
        kappaVals = kappaTildeVals * sqrt(c_o)
        sVals = zeros_like(kappaVals)
        extremalIndices = zeros_like(kappaVals)

        for (i, kappa) in enumerate(kappaVals):
            if(N):
                sVals[i], extremalIndices[i] = scenarioMethod(
                    kappa, c_i, c_o, N, removeResonances=removeResonances)
            else:
                raise Exception("Invalid arguments")
            # elif(n):
                # sVals[i] = scenarioMethod(kappa, c_i, c_o, n, index)
        print(sVals)
        return kappaTildeVals, sVals, extremalIndices

    def plotScenario(self, scenarioName, c_i, c_o,  n=None, N=None, plotRange=[2.0, 10.0],
                     index=None, plotBesselRoots=False, shiftFirstValue=True, removeResonances=False):

        scenarioMethodOf = {"getSingularValueOfBlock": self.getSingularValueOfBlock,
                            "ratioMaximumMinimumSingularValue": self.ratioMaximumMinimumSingularValue,
                            "MinimumSingularValue": self.getMinimumSingularValue,
                            "InvertedMinimumSingularValue": self.getInvertedMinimumSingularValue,
                            "MaximumSingularValue": self.getMaximumSingularValue,
                            "MaximumEuclidianNorm": self.getMaximumEuclidianNorm}

        yLabelNameOf = {"getSingularValueOfBlock": r"$\sigma$",
                        "ratioMaximumMinimumSingularValue": r"$\sigma_{max} / \sigma_{min}$",
                        "MinimumSingularValue": r"$\sigma_{min}$",
                        "InvertedMinimumSingularValue": r"$\frac{1}{\sigma_{min}}$",
                        "MaximumSingularValue": r"$\sigma_{max}$",
                        "MaximumEuclidianNorm": "Norm"
                        }

        scenarioMethod = scenarioMethodOf[scenarioName]

        plotName = self.getPlotName(scenarioName, c_i, c_o, n, N,
                                    index, plotBesselRoots, plotRange, shiftFirstValue, removeResonances)

        kappaTildeVals, sVals, selectedIndices = self.simulate(
            scenarioMethod, c_i, c_o, n, N, index, plotRange, removeResonances=removeResonances)
        # print(selectedIndices)
        print(selectedIndices.min())
        print(selectedIndices.max())
        # plt.plot(kappaTildeVals, selectedIndices)
        kappaTildeVals, sol, _ = self.simulate(
            self.getMaximumEuclidianNorm, c_i, c_o, n, N, index, plotRange)
        yLabelName = ""
        if(removeResonances):
            yLabelName = r"$\left\|A_{\mathcal{S}}^{-1}L_{\varepsilon}\right\|_{o p}$"
        else:
            yLabelName = r"$\left\|A_{\mathcal{S}}^{-1}\right\|_{o p}$"
        if(shiftFirstValue):
            yLabelName += r"$ -  y_0$"
        # yLabelNameOf[scenarioName]
        # print(selectedIndices)
        if(shiftFirstValue):
            yshift = sVals[0] - sol[0]
        else:
            yshift = 0
        plt.figure()
        plt.semilogy(kappaTildeVals, sol, label=r"$\|S\|$")
        self.plot(kappaTildeVals, sVals - yshift,
                  r"$\tilde\kappa$", yLabelName)

        if(plotBesselRoots):
            if(N == None):
                raise Exception("N needs to be defined to plot bessel roots")
            rescalePlotRange = array(plotRange) * sqrt(c_i) / sqrt(c_o)
            besselRoots = self.getBesselRootsFromInterval(
                rescalePlotRange[0], rescalePlotRange[1], N)
            maxVal = sVals.max()
            plt.vlines(besselRoots * sqrt(c_o) / sqrt(c_i), zeros_like(besselRoots),
                       maxVal * ones_like(besselRoots), linestyles='dashed')
        plt.savefig("./figures/" + self.modelName + "/" + plotName +
                    "indexrange_" + str(selectedIndices.min()) + "-" + str(selectedIndices.max()) + "_y_0_" + str(yshift) + ".pdf")

    # methods for plotting

    def plot(self, x, y, xLabelName, yLabelName):
        plt.semilogy(
            x, y, 'r', label=yLabelName)
        plt.xlabel(xLabelName)
        plt.ylabel("Operator Norm")
        plt.grid(True)
        plt.legend()

    def getPlotName(self, scenarioName, c_i, c_o, n, N, index,
                    plotBesselRoots, plotRange, shiftFirstValue, removeResonances):
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
        plotName += "shiftFirstValue_" + str(shiftFirstValue)
        plotName += "removeResonances_" + str(removeResonances)
        return plotName

    def convergenceTest(self, kappa, c_i, c_o, nRange):
        sVals = zeros_like(nRange, dtype=float)
        for (i, N) in enumerate(nRange):
            invertedSingularValue = float(1/(self.getSingularValueOfBlock(
                kappa, c_i, c_o, N).min()))
            sVals[i] = invertedSingularValue
        plt.figure()

        self.plot(nRange, sVals, r"$n$",
                  r"$\left\|(A^{num}_{n})^{-1}\right\|_{o p}$")
        plt.savefig("./figures/convergence_test/convergenceTest_c_i" +
                    str(c_o) + "c_o_" + str(c_o) + "kappa_" + str(kappa) + ".pdf")

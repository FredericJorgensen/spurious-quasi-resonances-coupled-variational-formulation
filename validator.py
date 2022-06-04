from matrix_model import MatrixModel
from boundary_conditions import BoundaryConditions
from numpy import *
from scipy.linalg import norm
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.special import jn_zeros, jv, jvp, hankel1, h1vp


class PValidator:
    def __init__(self, eta=1):
        self.model = MatrixModel(eta)
        self.boundaryConditions = BoundaryConditions(eta)
        self.eta = eta

    def P_3(self):
        return array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 1]])

    def composedMatrixBlock(self, kappa, c_i, c_o, n):
        A_inv = inv(self.model.getBlock(kappa, c_i, c_o, n))
        return self.P_3() @ A_inv @ self.boundaryConditions.P_b(kappa, c_i, c_o, n)

    def getNormOfComposedMatrix(self, kappa, c_i, c_o, N):
        acc = 0
        for n in range(-N, N + 1):
            composedMatrixBlock = self.composedMatrixBlock(kappa, c_i, c_o, n)
            acc += norm(composedMatrixBlock.flatten())
        return acc

    def simulate(self, c_i, c_o, plotRange=[4.0, 8.0], N=100):
        kappaVals = linspace(plotRange[0], plotRange[1], N)
        nVals = zeros_like(kappaVals)

        for (i, kappa) in enumerate(kappaVals):
            nVals[i] = self.getNormOfComposedMatrix(kappa, c_i, c_o, N)

        return kappaVals, nVals

    def plotValidator(self, c_i, c_o, plotRange=[4.0, 8.0], N=100):
        # it is a matrix with (2N + 1) blocks, so in total there are
        kappaVals, nVals = self.simulate(
            c_i, c_o, plotRange=plotRange, N=N)

        self.plot(kappaVals, nVals, r"$\kappa$", "Matrix Norm")

        plotName = self.getPlotName(c_i, c_o, N, plotRange)
        plt.savefig("./figures/p_validation/" + plotName + ".pdf")

    def plot(self, x, y, xLabelName, yLabelName):
        plt.figure()
        plt.semilogy(x, y, 'r')
        plt.xlabel(xLabelName)
        plt.ylabel(yLabelName)
        plt.grid(True)

    def getPlotName(self, c_i, c_o,  N,
                    plotRange):
        plotName = "validate_p_" + "c_i" + str(c_i) + "c_o" + str(c_o)
        if(N):
            plotName += "N_" + str(N)

        plotName += "plotRangeStart_" + \
            str(plotRange[0]) + "plotRangeEnd_" + str(plotRange[1])
        return plotName


class SimpleSolValidator:
    def __init__(self, eta=1):
        self.model = MatrixModel(eta)
        self.boundaryConditions = BoundaryConditions(eta)
        self.eta = eta

    def getBoundCoeffs(self, c_i, c_o, kappa, n):
        c_tilde = c_i / c_o
        f_1n = 2 * pi * (hankel1(n, kappa) - jv(n, kappa * sqrt(c_tilde)))
        f_2n = 2 * pi * (kappa * h1vp(n, kappa) - sqrt(c_tilde) *
                         kappa * jvp(n, kappa * sqrt(c_tilde)))
        return f_1n, f_2n

    def anaSol(self, c_i, c_o, kappa, n, N=100):
        c_tilde = c_i / c_o
        s1 = 1 / self.model.v(kappa, c_i, c_o, n) * jv(n, kappa * sqrt(c_tilde))
        s2 = 1 / self.model.w(kappa, c_i, c_o, n) * kappa * h1vp(n, kappa)
        s3 = 0

        baseLength = 2 * N + 1
        sol = zeros((3 * baseLength, ), dtype=complex)
        sol[3 * (n + N): 3 * (n + N + 1)] = array([s1, s2, s3])

        return sol

    def numSol(self, c_i, c_o, kappa, n,  N=100):
        A = self.model.getBlockMatrix(kappa, c_i, c_o, N)

        baseLength = 2 * N + 1
        f_1n = zeros((baseLength, ), dtype=complex)
        f_2n = zeros((baseLength, ), dtype=complex)
        f_1n_coeff, f_2n_coeff = self.getBoundCoeffs(c_i, c_o, kappa, n)
        f_1n[n + N] = f_1n_coeff
        f_2n[n + N] = f_2n_coeff
        b = self.boundaryConditions.b(kappa, c_i, c_o, f_1n, f_2n, N)
        return (linalg.solve(A, b)).flatten()

    def getResidualOfScenario(self, c_i, c_o, kappa, N=100):
        baseLength = 2 * N + 1
        vals = zeros((baseLength, ), )
        for index, n in enumerate(range(-N, N + 1)):
            anaSol = self.anaSol(c_i, c_o, kappa, n, N)
            numSol = self.numSol(c_i, c_o, kappa, n, N)
            residual = anaSol - numSol
            residualNorm = norm(residual) / norm(anaSol)
            vals[index] = residualNorm
        return(vals.max())

    def plotScenario(self, c_i, c_o, N=100, plotRange=[4.0, 8.0]):
        # plotRange is referring to kappa
        baseLength = N
        kappaVals = linspace(plotRange[0], plotRange[1], 100)
        residualVals = zeros_like(kappaVals)
        for (i, kappa) in enumerate(kappaVals):
            residualVals[i] = self.getResidualOfScenario(c_i, c_o, kappa, N)
            print("Completed i:", i, " of ", N)
            print("With result resVal: ", residualVals[i])

        plotName = self.getPlotName(c_i, c_o,  N,
                                    plotRange)

        savetxt("./figures/s_validation/" + plotName +
                "__residualVals.csv", residualVals, delimiter=",")

        self.plot(kappaVals, residualVals, r"$\kappa$", r"$\zeta(\kappa)$")
        plt.savefig("./figures/s_validation/" + plotName + ".pdf")

    def plot(self, x, y, xLabelName, yLabelName):
        plt.figure()
        plt.semilogy(x, y, 'r')
        plt.xlabel(xLabelName)
        plt.ylabel(yLabelName)
        plt.grid(True)

    def getPlotName(self, c_i, c_o,  N,
                    plotRange):
        plotName = "validate_sol_" + "c_i" + str(c_i) + "c_o" + str(c_o)
        if(N):
            plotName += "N_" + str(N)

        plotName += "plotRangeStart_" + \
            str(plotRange[0]) + "plotRangeEnd_" + str(plotRange[1])
        return plotName

from matrix_model import MatrixModel
from boundary_conditions import BoundaryConditions
from numpy import *
from scipy.linalg import norm
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.special import jn_zeros, jv, jvp, hankel1, h1vp


class PValidator:
    def __init__(self, eta=1):
        self.model = MatrixModel("regularised", eta)
        self.boundaryConditions = BoundaryConditions(eta)
        self.eta = eta

    def P_3(self):
        return array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 1]])

    def composedMatrixBlock(self, kappa, c_i, n):
        A_inv = inv(self.model.getA_TildeBlock(kappa, c_i, n))
        return self.P_3() @ A_inv @ self.boundaryConditions.P_b(kappa, c_i, n)

    def getNormOfComposedMatrix(self, kappa, c_i, N):
        acc = 0
        for n in range(-N, N + 1):
            composedMatrixBlock = self.composedMatrixBlock(kappa, c_i, n)
            acc += norm(composedMatrixBlock.flatten())
        return acc

    def simulate(self, c_i, plotRange=[4.0, 8.0], N=100):
        kappaVals = linspace(plotRange[0], plotRange[1], N)
        nVals = zeros_like(kappaVals)

        for (i, kappa) in enumerate(kappaVals):
            nVals[i] = self.getNormOfComposedMatrix(kappa, c_i, N)

        return kappaVals, nVals

    def plotValidator(self, c_i, plotRange=[4.0, 8.0], N=100):
        # it is a matrix with (2N + 1) blocks, so in total there are
        plotName = "plotTest"  # compute more sophisticated name here
        kappaVals, nVals = self.simulate(
            c_i, plotRange=plotRange, N=N)

        self.plot(kappaVals, nVals, r"$\kappa$", "Matrix Norm")

        plt.savefig("./figures/" + "/" + plotName + ".pdf")

    def plot(self, x, y, xLabelName, yLabelName):
        plt.figure()
        plt.plot(x, y, 'r')
        plt.xlabel(xLabelName)
        plt.ylabel(yLabelName)
        plt.show()


class SolValidator:
    def __init__(self, eta=1):
        self.model = MatrixModel("regularised", eta)
        self.boundaryConditions = BoundaryConditions(eta)
        self.eta = eta

    def solve(self, c_i, kappa, c_o, N=100):
        A = self.model.getA_Tilde(kappa, c_i, N)

        baseLength = 2 * N + 1
        f_1n = zeros((baseLength, ), dtype=complex)
        f_2n = zeros((baseLength, ), dtype=complex)
        kappa_i = kappa * c_i
        kappa_o = kappa * c_o
        f_1n[N] = 2 * pi * (hankel1(0, kappa_o) - jv(0, kappa_i))
        f_2n[N] = 2 * pi * (kappa_i * jv(1, kappa_i) - kappa_o * hankel1(1, kappa_o))
        b = self.boundaryConditions.b(kappa, c_i, f_1n, f_2n, N)

        sol = linalg.solve(A, b)
        Asel = A[3 * N: 3 * (N + 1), 3 * N: 3 * (N + 1)]
        bsel = b[3 * N: 3 * (N + 1)]
        sol = sol[3 * N: 3 * (N + 1)]
        set_printoptions(precision=2)
        print("Asel:", Asel )
        print("bsel:", bsel)
        print("sol:", sol * self.model.v(kappa, c_i, 0))

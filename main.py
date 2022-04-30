from simulator import Simulator
from validator import PValidator, SolValidator


if __name__ == "__main__":
    #pvalidator = PValidator(eta=1)
    #pvalidator.plotValidator(1.0, plotRange=[4.0, 8.0], N=100)
    solValidator = SolValidator(eta=1)
    #c_i, kappa, c_o, N = 100):
    solValidator.solve(1.0, 5.0, 1.0, N=100)
    #unregularisedSimulator = Simulator("unregularised")

    # unregularisedSimulator.plotScenario("MinimumSingularValue", 1.0, N=100, plotRange=[
    #                                   4.0, 8.0])
    # regularisedSimulator = Simulator("regularised", eta = 1.0)

    # regularisedSimulator.plotScenario("MinimumSingularValue", 3.0, N=100, plotRange=[
    #                                        4.0, 8.0])
    # #for eta in [1, 100, 1000]:
    #regularisedSimulator = Simulator("regularised", eta = eta)

    # regularisedSimulator.plotScenario("MinimumSingularValue", 3.0, N=100, plotRange=[
    #                                   4.0, 8.0])

    # regularisedSimulator.plotScenario("MaximumSingularValue", 3.0, N=100, plotRange=[
    # 4.0, 8.0])

    # regularisedSimulator.plotScenario("MinimumSingularValue", 3.0, N=100, plotRange=[
    # 4.0, 8.0])

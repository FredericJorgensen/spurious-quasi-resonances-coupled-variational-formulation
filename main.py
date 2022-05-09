from simulator import Simulator
from validator import PValidator, SimpleSolValidator


if __name__ == "__main__":

    scenario1 = [1.0, 3.0]
    scenario2= [3.0, 1.0]

    #parameters
    scenario = scenario1
    c_i = scenario[0]
    c_o = scenario[1]
    kappa = 1
    n = 1
    kappaRange = [4.0, 8.0]
    N = 100


    #pvalidator = PValidator(eta=1)
    #pvalidator.plotValidator(c_i, c_o, plotRange=kappaRange, N=N)

    simpleSolValidator = SimpleSolValidator(eta=1)
    #c_i, kappa, c_o, N = 100):
    simpleSolValidator.plotScenario(c_i, c_o, N=100, plotRange=[5.5, 5.6])

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

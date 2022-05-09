from simulator import Simulator
from validator import PValidator, SimpleSolValidator

if __name__ == "__main__":

    scenario1 = [1.0, 3.0]
    scenario2 = [3.0, 1.0]

    # parameters
    scenario = scenario2
    c_i = scenario[0]
    c_o = scenario[1]
    kappa = 1
    n = 1
    kappaRange = [0.5, 15.0]
    N = 100
    eta = 1.0

    # simulate simple sol:
    #simpleSolValidator = SimpleSolValidator(eta=1)
    #simpleSolValidator.plotScenario(c_i, c_o, N=100, plotRange=kappaRange)
    #simpleSolValidator.plotScenario(c_o, c_i, N=100, plotRange=kappaRange)

    # kernel check:
    #pvalidator = PValidator(eta=1)
    #pvalidator.plotValidator(c_i, c_o, plotRange=kappaRange, N=N)

    # numerical simulation
    regularisedSimulator = Simulator("regularised", eta=eta)
    regularisedSimulator.plotScenario(
        "ratioMaximumMinimumSingularValue", c_i, c_o, N=100,  plotBesselRoots=True,
        plotRange=kappaRange)


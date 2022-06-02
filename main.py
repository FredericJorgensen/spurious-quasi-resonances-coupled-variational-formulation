from simulator import Simulator
from validator import PValidator, SimpleSolValidator


def simulateScenario(selectedScenario):
    assert(type(selectedScenario) == int and 1 <= selectedScenario
           and 6 >= selectedScenario)
    scenario1 = [1.0, 3.0, 1.0]
    scenario2 = [3.0, 1.0, 1.0]
    scenario3 = [1.0, 10.0, 1.0]
    scenario4 = [10.0, 1.0, 1.0]
    scenario5 = [1.0, 3.0, 100.0]
    scenario6 = [1.0, 3.0, 10]

    scenarios = [scenario1, scenario2, scenario3, scenario4, scenario5,
                 scenario6]


    scenario = scenarios[selectedScenario - 1]

    # define parameters
    c_i = scenario[0]
    c_o = scenario[1]
    eta = scenario[2]
    kappaRange = [0.5, 15.0]
    N = 100

    simulator = Simulator(model="variational", eta=eta)
    simulator.plotScenario(
        "InvertedMinimumSingularValue", c_i, c_o, N=100,  plotBesselRoots=False,
        plotRange=kappaRange)


def validateSimpleSol(selectedScenario):
    assert(type(selectedScenario) == int and 1 <= selectedScenario
           and 2 >= selectedScenario)

    scenario1 = [1.0, 3.0, 1.0]
    scenario2 = [3.0, 1.0, 1.0]
    scenarios = [scenario1, scenario2]
    scenario = scenarios[selectedScenario - 1]

    # define parameters
    c_i = scenario[0]
    c_o = scenario[1]
    eta = scenario[2]
    kappaRange = [0.5, 15.0]
    N = 100

    simpleSolValidator = SimpleSolValidator(eta=eta)
    simpleSolValidator.plotScenario(c_i, c_o, N=N, plotRange=kappaRange)


def validateProjector(selectedScenario):
    assert(type(selectedScenario) == int and 1 <= selectedScenario
           and 2 >= selectedScenario)

    scenario1 = [1.0, 3.0, 1.0]
    scenario2 = [3.0, 1.0, 1.0]
    scenarios = [scenario1, scenario2]
    scenario = scenarios[selectedScenario - 1]
    c_i = scenario[0]
    c_o = scenario[1]
    eta = scenario[2]
    kappaRange = [0.5, 15.0]
    N = 100

    pvalidator = PValidator(eta=eta)
    pvalidator.plotValidator(c_i, c_o, plotRange=kappaRange, N=N)


if __name__ == "__main__":
    simulateScenario(4)
    simulateScenario(5)
    simulateScenario(6)
    validateSimpleSol(2)
    validateProjector(2)


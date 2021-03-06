from simulator import Simulator
from validator import PValidator, SimpleSolValidator
from numpy import *
import matplotlib.pyplot as plt
from matrix_model import MatrixModel

# This file should be used for calling methods to perform calculations & simulations and generate plots

# method to plot the operator norm of the inverse regularised operator (as introduced in section 4) as a function for kappa.
# used to generate plots in section 7 and 9
def simulateScenario(selectedScenario):
    assert(type(selectedScenario) == int and 1 <= selectedScenario
           and 13 >= selectedScenario)
    scenario1 = [1.0, 3.0, 1.0, True, False, 1]
    scenario2 = [3.0, 1.0, 1.0, True, False, 1]
    scenario3 = [1.0, 10.0, 1.0, True, False, 1]
    scenario4 = [10.0, 1.0, 1.0, True, False, 1]
    scenario5 = [1.0, 3.0, 100.0, True, False, 1]
    scenario6 = [1.0, 3.0, 10, True, False, 1]
    scenario7 = [1.0, 3.0, 5.0, True, False, 1]
    scenario8 = [1.0, 3.0, 0.0, True, False, 1]
    scenario9 = [1.0, 3.0, 0.5, True, False, 1]
    scenario10 = [1.0, 3.0, 1.0, False, True, 1]
    scenario11 = [3.0, 1.0, 1.0, False, True, 1]
    scenario12 = [3.0, 1.0, 1.0, True, False, 30]
    scenario13 = [1.0, 3.0, 1.0, True, False, 30]


    scenarios = [scenario1, scenario2, scenario3, scenario4, scenario5,
                 scenario6, scenario7, scenario8, scenario9, scenario10, scenario11, scenario12, scenario13]

    scenario = scenarios[selectedScenario - 1]

    print("starting simulation inverse augmented operator norms. scenario: ", selectedScenario)


    # define parameters of Helmholtz problem & Fourier cutoff
    c_i = scenario[0]
    c_o = scenario[1]
    eta = scenario[2]
    shiftFirstValue = scenario[3]
    removeResonances = scenario[4]
    scaleLastEquation =scenario[5]

    kappaTildeRange = [0.5, 15]
    N = 30

    # configure simulation with regularising parameter eta as in section 4
    simulator = Simulator(eta=eta)

    # perform simulation
    simulator.plotScenario(
        "simulation_scenario_" + str(selectedScenario), c_i, c_o, N=N,  plotBesselRoots=False,
        plotRange=kappaTildeRange, shiftFirstValue = shiftFirstValue, removeResonances=removeResonances, 
        scaleLastEquation = scaleLastEquation)

# method to plot the convergence of the operator norm of the inverse regularised operator as a function of the Fourier mode
# used to generate plots in section 9
def convergenceTest(selectedScenario):
    assert(type(selectedScenario) == int and 1 <= selectedScenario
           and 7 >= selectedScenario)
    scenario1 = [1.0, 3.0, 1.0]
    scenario2 = [3.0, 1.0, 1.0]
    scenario3 = [1.0, 10.0, 1.0]
    scenario4 = [10.0, 1.0, 1.0]
    scenario5 = [1.0, 3.0, 100.0]
    scenario6 = [1.0, 3.0, 10]
    scenario7 = [1.0, 3.0, 5.0]

    scenarios = [scenario1, scenario2, scenario3, scenario4, scenario5,
                 scenario6, scenario7]

    scenario = scenarios[selectedScenario - 1]

    # define parameters of Helmholtz problem & Fourier cutoff    
    c_i = scenario[0]
    c_o = scenario[1]
    eta = scenario[2]
    nRange = range(1, 50)
    kappa = 1.0 

    simulator = Simulator(eta=eta)
    simulator.convergenceTest(kappa, c_i, c_o, nRange)

# method to validate the Galerkin matrix of the regularised operator by comparing its numerical to its analytical solution
# used to generate plots in section 6
def validateSimpleSol(selectedScenario):
    assert(type(selectedScenario) == int and 1 <= selectedScenario
           and 2 >= selectedScenario)

    scenario1 = [1.0, 3.0, 1.0]
    scenario2 = [3.0, 1.0, 1.0]
    scenarios = [scenario1, scenario2]
    scenario = scenarios[selectedScenario - 1]

    # define parameters of Helmholtz problem & Fourier cutoff    
    c_i = scenario[0]
    c_o = scenario[1]
    eta = scenario[2]
    kappaTildeRange = [0.5, 15.0]
    N = 30
    numberOfValues = 50

    simpleSolValidator = SimpleSolValidator(eta=eta)
    simpleSolValidator.plotScenario(c_i, c_o, N=N, plotRange=kappaTildeRange,
                                    numberOfValues=numberOfValues)


# method to validate the Galerkin matrix of the regularised operator by checking whether p = 0 is satisfied for the solution
# used to generate plots in section 6
def validateProjector(selectedScenario):
    assert(type(selectedScenario) == int and 1 <= selectedScenario
           and 2 >= selectedScenario)

    scenario1 = [1.0, 3.0, 1.0]
    scenario2 = [3.0, 1.0, 1.0]
    scenarios = [scenario1, scenario2]
    scenario = scenarios[selectedScenario - 1]

    # define parameters of Helmholtz problem & Fourier cutoff    
    c_i = scenario[0]
    c_o = scenario[1]
    eta = scenario[2]
    kappaTildeVals = [0.5, 15.0]
    N = 100

    pvalidator = PValidator(eta=eta)
    pvalidator.plotValidator(c_i, c_o, plotRange=kappaTildeVals, N=N)

# this method includes the simulations required to generate all plots used in the paper
def paperSimulations():
    #print("starting numerical solution validation for scenario 1 (section 6)")
    #validateSimpleSol(1)
    # print("starting numerical solution validation for scenario 2 (section 6)")
    # validateSimpleSol(2)
    # print("starting p = 0 validation for scenario 1 (section 6)")
    # validateProjector(1)
    # print("starting p = 0 validation for scenario 2 (section 6)")
    # validateProjector(2)
    # print("starting simulation inverse operator norms (section 7, 9)")
    # simulateScenario(1)
    # simulateScenario(2)
    # simulateScenario(3)
    # simulateScenario(4)
    # simulateScenario(5)
    # simulateScenario(6)
    # simulateScenario(7)
    # simulateScenario(8)
    # simulateScenario(9)
    simulateScenario(12)
    # simulateScenario(11)
    # simulateScenario(12)
    #simulateScenario(13)

   
    #print("starting convergence test of inverse augmented operator norms (section 9)")
    #convergenceTest(4)

if __name__ == "__main__":
    paperSimulations()
from simulator import Simulator


if __name__ == "__main__":
    unregularisedSimulator = Simulator("unregularised")

    unregularisedSimulator.plotScenario("ratioMaximumMinimumSingularValue", 3.0, N=100, plotRange=[
                                        6.8, 6.82], plotBesselRoots=True)


    regularisedSimulator = Simulator("unregularised")

    regularisedSimulator.plotScenario("ratioMaximumMinimumSingularValue", 3.0, N=100, plotRange=[
                                        4.0, 8.0], plotBesselRoots=True)


from simulator import Simulator


if __name__ == "__main__":
    #unregularisedSimulator = Simulator("unregularised")

    #unregularisedSimulator.plotScenario("MinimumSingularValue", 1.0, N=100, plotRange=[
     #                                   4.0, 8.0])
    regularisedSimulator = Simulator("regularised", eta = 1.0)
    
    regularisedSimulator.plotScenario("MinimumSingularValue", 3.0, N=100, plotRange=[
                                           4.0, 8.0])
    #for eta in [1, 100, 1000]:
        #regularisedSimulator = Simulator("regularised", eta = eta)
    
        #regularisedSimulator.plotScenario("MinimumSingularValue", 3.0, N=100, plotRange=[
         #                                   4.0, 8.0])
    
        #regularisedSimulator.plotScenario("MaximumSingularValue", 3.0, N=100, plotRange=[
                                           # 4.0, 8.0])
    
        #regularisedSimulator.plotScenario("MinimumSingularValue", 3.0, N=100, plotRange=[
                                           # 4.0, 8.0])


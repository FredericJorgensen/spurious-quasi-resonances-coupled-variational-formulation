from matrix_model import MatrixModel

### validate that p is always an element of the null space
def testP(model, eta, c_i, plotRange = [4.0, 8.0]):
    model = MatrixModel(model, eta)

    model.getA_Tilde(kappa, c_i, n)
from sympy import Matrix, symbols, init_printing, pprint, sqrt
import numpy as np
import matplotlib.pyplot as plt
from utils import lambdaK
init_printing()

W, K, e = symbols('W K e')

AI = Matrix([[-1, -W, 0], [0, K-1/2, -W], [0, W, e]])
pprint(AI.inv())

ns = np.linspace(0, 50, 1000)
kappa = 5.0
vals = [lambdaK(n, kappa) for n in ns]

plt.figure()
plt.plot(ns, vals)


# prove linear growth of coefficients (also use WolframAlpha)
# n= symbols('n')
# U = Matrix([[1 / (n**2), 1 / sqrt(n)], [1 / sqrt(n), 1]])

# pprint(U.eigenvals())


# def f(n):
#     a = (n ** 2 + 1) / (2 * n ** 2)
#     b = np.sqrt(n ** 4 + 4 * n ** 3 - 2 * n ** 2 + 1) / (2 * n ** 2)
#       # b = np.sqrt(n ** 4 + 4 * n ** 3 - 2 * n ** 2 + 1) / (2 * n ** 2)

#     return a - b

# ns = np.linspace(1, 100, 100)
# vals = np.zeros_like(ns)

# for i, n in enumerate(ns):
#     vals[i] = 1 / f(n)

# plt.figure()
# plt.plot(ns, vals)
# plt.show()

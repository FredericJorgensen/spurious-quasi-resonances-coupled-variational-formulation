from sympy import Matrix, symbols, init_printing, pprint, sqrt
import numpy as np
import matplotlib.pyplot as plt
init_printing()

# Ki, Ko, Vi, Vo, Wi, Wo, Kip, Kop = symbols('Ki Ko Vi Vo Wi Wo Kip Kop')

# AI = Matrix([[-(Ki + Ko), Vi + Vo], [Wi + Wo, Kip + Kop]])
# Po = Matrix([[1/2 - Ko, Vo], [Wo, 1/2 + Kop]])
#pprint(AI.inv() @ Po)

# prove linear growth of coefficients (also use WolframAlpha)
n= symbols('n')
U = Matrix([[1 / (n**2), 1 / sqrt(n)], [1 / sqrt(n), 1]])

pprint(U.eigenvals())



def f(n):
    a = (n ** 2 + 1) / (2 * n ** 2)
    b = np.sqrt(n ** 4 + 4 * n ** 3 - 2 * n ** 2 + 1) / (2 * n ** 2)
      # b = np.sqrt(n ** 4 + 4 * n ** 3 - 2 * n ** 2 + 1) / (2 * n ** 2)

    return a - b

ns = np.linspace(1, 100, 100)
vals = np.zeros_like(ns)

for i, n in enumerate(ns):
    vals[i] = 1 / f(n)

plt.figure()
plt.plot(ns, vals)
plt.show()
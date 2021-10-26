import numpy as np
import matplotlib.pyplot as plt


x0 = 2.0
func = lambda x: np.exp(-0.5 * np.power(x - x0, 2)) * (x<=0)

xx = np.linspace(-3, 0, 101)
ff = func(xx)

plt.figure()
plt.plot(xx, ff)

all_orders = [1,2,3,4]
for order in all_orders:
    kk = np.arange(order)
    jj = np.arange(order) + 1
    M = np.power(-jj[None,:], kk[:,None])
    mu = np.linalg.solve(M, np.ones(order))

    yy = np.linspace(0, 3, 101)
    gg = np.zeros(len(yy))
    for i in range(len(yy)):
        y = yy[i]
        gg[i] = np.sum(func(-jj * y) * mu)

    plt.plot(yy,gg)
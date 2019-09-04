import numpy as np


def ou_noise(n, sigma = 2., tau=5. ):
    mu = 0
    sigma_bis = sigma * np.sqrt(2. / tau)

    x = np.zeros(n)
    x[0] = np.random.randn() * sigma
    for i in range(n-1):
        x[i + 1] = x[i] + (-(x[i] - mu) / tau) + sigma_bis * np.random.randn()
    return x
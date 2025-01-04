# Ramachandran S - 23b1052 
# Keyaan KR - 23b0977 
# Harith S - 23b1085

#This code is for question 2c.

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def sample(loc, scale, N=100000):
    uniform_samples = np.random.uniform(0, 1, N)
    return norm.ppf(uniform_samples, loc=loc, scale=scale)


def plot(params, N=100000):
    plt.figure(figsize=(10, 6))
    colors = ["blue", "red", "yellow", "green"]

    for (mean, var), color in zip(params, colors):
        std = np.sqrt(var)
        samples = sample(mean, std, N)
        plt.hist(
            samples,
            bins=500,
            density=True,
            alpha=0.5,
            color=color,
            label=f"μ={mean}, σ²={var}",
        )

    plt.title("Gaussian Distributions")
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("2c.png")
    plt.show()


params = [(0, 0.2), (0, 1.0), (0, 5.0), (-2, 0.5)]
plot(params)

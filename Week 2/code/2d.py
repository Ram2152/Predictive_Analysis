# Ramachandran S - 23b1052 
# Keyaan KR - 23b0977 
# Harith S - 23b1085

#This code is for question 2d.

import numpy as np
import matplotlib.pyplot as plt


def simulate(N, h):
    # generates random left/right moves for each ball at each level
    moves = np.random.randint(0, 2, size=(N, h))
    # converts left to decrement and right to increment
    steps = 2 * moves - 1
    # computes the final position of each ball by adding up the increments/decrements
    final_pos = np.sum(steps, axis=1)
    # counts the number of balls at each final position
    pos, counts = np.unique(final_pos, return_counts=True)
    # creates a array with all possible positions
    all_pos = np.arange(-h, h + 1)
    # creates a array with zeros and puts the counts at the correct positions
    distribution = np.zeros(2 * h + 1, dtype=float)
    # calculates the indices for each position
    indices = pos + h
    # updates the normalized counts at the correct positions
    distribution[indices] = counts / N

    return all_pos, distribution


def plot(positions, distribution, h, N, filename):
    plt.figure(figsize=(12, 6))
    plt.bar(positions, distribution, align="center", alpha=0.5)
    plt.title(f"Galton Board Simulation (h={h}, N={N})")
    plt.xlabel("Pocket")
    plt.ylabel("Normalized count")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


N = 100000

for i, h in enumerate([10, 50, 100], start=1):
    positions, distribution = simulate(N, h)
    plot(positions, distribution, h, N, f"2d{i}.png")

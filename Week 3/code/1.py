import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset (assuming you've already downloaded and filtered the data)
data = pd.read_csv('data.csv')
data = data.head(1500)  # Use the first 1500 data points
data = data[data['D (Mpc)'] < 4]  # Filter for distances less than 4 Mpc
filtered_data = data['D (Mpc)'].values

# Calculate the range of data
data_range = np.max(filtered_data) - np.min(filtered_data)

# Define bin edges with a width of 0.4, from 0 to 4
bins = np.arange(0, 4 + 0.4, 0.4)  # The '4 + 0.4' ensures the final bin reaches exactly 4

# Part (a): Plot histogram with 10 bins
plt.hist(filtered_data, bins=bins, edgecolor='black', alpha=0.7)
plt.title('Histogram of Distance (Mpc)')
plt.xlabel('Distance (Mpc)')
plt.ylabel('Frequency')
plt.savefig('10binhistogram.png')
plt.show()

# Calculate estimated probabilities for bins
counts, bin_edges = np.histogram(filtered_data, bins=bins)
total_counts = counts.sum()
estimated_probabilities = counts / total_counts
print('Estimated probabilities for bins:', estimated_probabilities)

# Part (c): Find optimal h

# Initialize variables
n = len(filtered_data)
m_values = np.arange(1, 1001)  # Number of bins from 1 to 1000
cv_scores = []
h_values = []

for m in m_values:
    # Compute bin width h
    h = 4 / m
    h_values.append(h)

    # Create histogram and get the counts in each bin
    hist, bin_edges = np.histogram(filtered_data, bins=m)

    # Calculate p_j (probability estimates for each bin)
    p_j = hist / n

    # Calculate the first term of J(h) (for cross-validation score)
    first_term = 2 / ((n - 1) * h)

    # Calculate the second term of J(h) using p_j
    second_term = (n + 1) / ((n - 1) * h) * np.sum(p_j**2)

    # Compute the cross-validation score for current bin count m
    J_h = first_term - second_term
    cv_scores.append(J_h)

# Plot the cross-validation scores vs number of bins
plt.figure(figsize=(8, 6))
plt.plot(h_values, cv_scores, label='Cross-Validation Score')
plt.xlabel('Number of Bins (m)')
plt.ylabel('Cross-Validation Score')
plt.title('Cross-Validation Score vs Number of Bins')
plt.grid(True)
plt.legend()
plt.savefig('crossvalidation.png')
# plt.show()

# Find the optimal number of bins (corresponding to minimum CV score)
optimal_m = m_values[np.argmin(cv_scores)]
optimal_h = 4 / optimal_m

# Now plot the histogram for the optimal bin width
plt.figure(figsize=(8, 6))
plt.hist(filtered_data, bins=optimal_m, edgecolor='black', alpha=0.7)
plt.title(f'Optimal Histogram (bins = {optimal_m})')
plt.xlabel('Distance (Mpc)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('optimalhistogram.png')
plt.show()

print(f"Optimal number of bins: {optimal_m}")
print(f"Optimal bin width (h*): {optimal_h}")

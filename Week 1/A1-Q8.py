import numpy as np
import matplotlib.pyplot as plt


def shift_image(img, tx):
    img_shifted = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # shift the image
            if j + tx < img.shape[1] and j + tx >= 0:
                img_shifted[i, j + tx] = img[i, j]

    return img_shifted


img = plt.imread("Mona_Lisa.jpeg")

# calculate to plot the normalized histogram
flatten_img = img.flatten()
normalized_img = flatten_img / img.size

# Plot histogram of the normalized image
plt.figure(figsize=(10, 6))
plt.hist(normalized_img, bins=50, edgecolor="black", color="skyblue", alpha=0.7)
plt.xlabel("Normalized Pixel Value", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.title("Histogram of Normalized Pixel Values", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.6)

# Save and show the plot
plt.tight_layout()
plt.savefig("histogram.png")
plt.show()
plt.close()

# Calculate the correlation coefficient for different shifts
shifts = range(-10, 11, 1)
correlations = []

for tx in shifts:
    if tx != 0:
        img_shifted = shift_image(img, tx)
        # Calculate correlation coefficient
        correlation = np.corrcoef(img.flatten(), img_shifted.flatten())[0, 1]
        correlations.append(correlation)

# Plot correlation coefficient vs shift
plt.figure(figsize=(10, 6))
plt.plot(shifts[1:], correlations, "ro-", markersize=4)  # Skip zero shift
plt.xlabel("Shift")
plt.ylabel("Correlation Coefficient")
plt.title("Correlation Coefficient vs Shift")
plt.savefig("correlation_vs_shift.png")
plt.show()
plt.close()

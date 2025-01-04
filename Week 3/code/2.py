import numpy as np
import matplotlib.pyplot as plt
import math

# Custom Epanechnikov KDE class
class EpanechnikovKDE:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.data = None

    def fit(self, data):
        """Fit the KDE model with the given data."""
        self.data = np.array(data)
        

    def epanechnikov_kernel(self, x, xi):
        """Epanechnikov kernel function."""

        # x is a point, xi is a set of points, so it broadcasts and computes the norm parallely
        # it returns an array of densities wrt to the given points

        distance = np.linalg.norm(x - xi, axis = 1)

        for index in range(len(distance)):
            if (distance[index] > 1): 
                distance[index] = 0
            else:
                # added the normalising factor
                distance[index] = (2/math.pi)*(  1 - distance[index])
        return distance

    def evaluate(self, x):
        """Evaluate the KDE at point x."""
        h = self.bandwidth
        n = len(self.data)
        
        
        prob_density = []

        density = np.sum(self.epanechnikov_kernel(x, self.data))
            
        

        density = density / (n * h * h)
        prob_density.append((x[0][0], x[0][1], density))
        return np.array(prob_density)
        


# Load the data from the NPZ file
data_file = np.load('transaction_data.npz')
data = data_file['data']


# TODO: Initialize the EpanechnikovKDE class
kernel = EpanechnikovKDE(0.3)

# TODO: Fit the data
kernel.fit(data)

x = np.linspace(-6,6,100)
y = np.linspace(-6,6,100)

size = len(x)
shape = (size, size)

# creates a mesh type structure

x, y = np.meshgrid(x,y)

points = np.zeros(((size**2),2))

for index_x in range(size):
    for index_y in range(size):
        points[index_x * size + index_y] = np.array((x[index_x][index_y], y[index_x][index_y]))


prob = np.array([kernel.evaluate(point.reshape(1,-1)) for point in points])
prob = prob.reshape(size*size, 3)
print(prob.shape)
# TODO: Plot the estimated density in a 3D plot
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Density Estimate')

# Set the title of the plot
ax.set_title('Epanechnikov KDE 23B1085-23B1052-23B0977')

ax.plot_surface(x, y, prob[:, 2].reshape(size, size),  cmap='viridis', edgecolor='none')
# TODO: Save the plot

plt.savefig("transaction distribution.png")

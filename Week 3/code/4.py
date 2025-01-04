import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# the following imports are used to read the data from the internet

import requests
from io import StringIO
import warnings

# Reading the data from the internet

# suppress warnings
warnings.filterwarnings('ignore')

# get the data
url = 'https://www.stat.cmu.edu/~larry/all-of-statistics/=data/glass.dat'

# verify = false is used to ignore SSL certificate verification
response = requests.get(url, verify=False)

# read the data, skipping the first row (which contains column names)
data = pd.read_csv(StringIO(response.text),
                   delim_whitespace=True, header=None, skiprows=1)
columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
data.columns = columns

# Naradaya-Watson Kernel Regression


def gaussian_kernel(values):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (values ** 2))


def epanechnikov_kernel(values):
    # create a mask for valid values
    valid_mask = np.abs(values) <= 1
    # create an array of zeros with the same size as the input values
    kernel_values = np.zeros(values.size)
    # set the valid values to the kernel values
    kernel_values[valid_mask] = 0.75 * (1 - values[valid_mask] ** 2)
    return kernel_values


def naradaya_watson(Y, X, center, h, kernel='gaussian'):

    kernel_weights = []
    if kernel == 'gaussian':
        kernel_weights = gaussian_kernel((center - X) / h)
    elif kernel == 'epanechnikov':
        kernel_weights = epanechnikov_kernel((center - X) / h)

    # if the sum of the kernel weights, to avoid division by zero
    if (np.sum(kernel_weights) == 0):
        return 0

    return np.sum(Y * kernel_weights) / np.sum(kernel_weights)


def cross_validation(X, Y, h, kernel='gaussian'):
    squared_errors = []
    for i in range(X.size):

        # set to be trained on
        X_train = np.delete(X, i)
        Y_train = np.delete(Y, i)

        # set to be tested on
        X_test = X[i]
        Y_test = Y[i]

        # predict Y value and calculate squared error
        Y_pred = naradaya_watson(Y_train, X_train, X_test, h, kernel)
        squared_errors.append((Y_pred - Y_test) ** 2)

    return np.mean(squared_errors)


def find_optimal_h(X, Y, kernel='gaussian'):
    # h values to be tested
    h_values = np.linspace(0.01, 1.00, 100)
    # array to store squared errors
    squared_errors = [cross_validation(X, Y, h, kernel) for h in h_values]
    # optimized h is the one that minimizes the squared errors
    return h_values[np.argmin(squared_errors)], squared_errors


# skips data points with Y value as zero, as they are invalid
def skip_plot_data(X, Y):
    valid_mask = Y != 0
    X = X[valid_mask]
    Y = Y[valid_mask]
    return X, Y


def plot_data(X, Y, kernel='gaussian'):

    optimized_h, squared_errors = find_optimal_h(X, Y, kernel)
    # undersmoothing
    u_smoothing_h = 0.01
    # oversmoothing
    o_smoothing_h = 1.00

    # plot the data
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.scatter(X, Y, s=40, c='blue', alpha=0.40)
    X_test = np.linspace(min(X), max(X), X.size)
    Y_pred = np.array([naradaya_watson(Y, X, center, o_smoothing_h, kernel)
                       for center in X_test])

    X_test, Y_pred = skip_plot_data(X_test, Y_pred)

    plt.plot(X_test, Y_pred, c='red')
    plt.title('Oversmoothed (h = {})'.format(o_smoothing_h))
    plt.grid(True)
    plt.tight_layout()

    plt.subplot(2, 2, 2)
    plt.scatter(X, Y, s=40, c='blue', alpha=0.40)
    X_test = np.linspace(min(X), max(X), X.size)
    Y_pred = np.array([naradaya_watson(Y, X, center, u_smoothing_h, kernel)
                       for center in X_test])

    X_test, Y_pred = skip_plot_data(X_test, Y_pred)

    plt.plot(X_test, Y_pred, c='red')
    plt.title('Undersmoothed (h = {})'.format(u_smoothing_h))
    plt.grid(True)
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    plt.scatter(X, Y, s=40, c='blue', alpha=0.40)
    X_test = np.linspace(min(X), max(X), X.size)
    Y_pred = np.array([naradaya_watson(Y, X, center, optimized_h, kernel)
                       for center in X_test])

    X_test, Y_pred = skip_plot_data(X_test, Y_pred)
    X_test, Y_pred = skip_plot_data(X_test, Y_pred)

    plt.plot(X_test, Y_pred, c='red')
    plt.title('Optimal bandwidth (h = {})'.format(optimized_h))
    plt.grid(True)
    plt.tight_layout()

    plt.subplot(2, 2, 4)
    h_values = np.linspace(0.01, 1.00, 100)
    plt.plot(h_values, squared_errors, c='blue')
    plt.axvline(x=optimized_h, c='red', linestyle='--')
    plt.axhline(y=squared_errors[np.argmin(squared_errors)],
                c='red', linestyle='--')
    plt.scatter(optimized_h, squared_errors[np.argmin(
        squared_errors)], color='blue', s=40, alpha=0.40,
        label='Optimal bandwidth = {}'.format(optimized_h))
    plt.xlim(0.00, 1.00)
    plt.legend()
    plt.title('LOOCV MSE vs h')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('{}_kernel_regression.png'.format(kernel), dpi=300)
    plt.close()


# read the data
RI = data['RI'].to_numpy()
Al = data['Al'].to_numpy()

plot_data(Al, RI, 'gaussian')
plot_data(Al, RI, 'epanechnikov')

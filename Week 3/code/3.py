import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Function to generate polynomial features
def polynomial_features(X, degree):
    X_poly = np.ones((X.shape[0], 1))  # Start with a column of ones for the intercept
    for d in range(1, degree + 1):
        X_poly = np.hstack((X_poly, X**d))
    return X_poly

# OLS function to compute the best fit parameters
def OLS(X, y):
    # Compute the OLS estimate of beta: (X^T X)^-1 X^T y
    XtX_inv = np.linalg.inv(X.T.dot(X))
    beta = XtX_inv.dot(X.T).dot(y)
    return beta

# Function to make predictions
def predict(X, beta):
    return X.dot(beta)

# Load your training data from CSV
train_data = pd.read_csv('train.csv')  # Make sure the file is in the correct directory
X_train_orig = train_data.iloc[:, 1].values.reshape(-1, 1)  # 2nd column as X_train
y_train_orig = train_data.iloc[:, 2].values  # 3rd column as Y_train

# Load your test data from CSV
test_data = pd.read_csv('test.csv')  # Make sure the file is in the correct directory
X_test = test_data.iloc[:, 1].values.reshape(-1, 1)  # 2nd column as X_test

# Let's assume we are working with degree 3 polynomials
degree = 3

# Generate polynomial features
X_poly_train = polynomial_features(X_train_orig, degree)

# Fit the model
beta = OLS(X_poly_train, y_train_orig)

# Save the learned weights (beta) as a pickle file
with open('3_weights.pkl', 'wb') as f:
    pickle.dump(beta, f)

# Generate polynomial features for the test set
X_poly_test = polynomial_features(X_test, degree)

# Predict on the test set
y_pred = predict(X_poly_test, beta)

# Add the predicted y values as a new column to the test dataframe
test_data['y'] = y_pred

# Save the updated dataframe to a new CSV file with the predictions
test_data.to_csv('3_predictions.csv', index=False)

# Print the learned parameters
print("Learned parameters (beta):", beta)

#---------------------- 3.3.a done ----------------------#

# Split the training data into training and validation sets (90:10)
split_index = int(len(X_train_orig) * 0.9)
X_dev = X_train_orig[split_index:]  # Development set
y_dev = y_train_orig[split_index:]
X_train = X_train_orig[:split_index]  # Remaining for training
y_train = y_train_orig[:split_index]

# Degrees to evaluate
degrees = list(range(1, 51))
mse_dev = []

# Train models and calculate MSE
for degree in degrees:
    # Generate polynomial features
    X_poly_train = polynomial_features(X_train, degree)
    X_poly_dev = polynomial_features(X_dev, degree)

    # Fit the model
    beta = OLS(X_poly_train, y_train)

    # Predict on  development set
    y_dev_pred = predict(X_poly_dev, beta)

    # Calculate Mean Squared Error
    dev_error = np.mean((y_dev - y_dev_pred) ** 2)

    mse_dev.append(dev_error)

# Identify underfit, correct fit, and overfit
correct_fit_degree = degrees[mse_dev.index(min(mse_dev))]
underfit_degree = max(2, correct_fit_degree - 3)
overfit_degree = min(50, correct_fit_degree + 10)

print(f"Underfit Degree: {underfit_degree}, Correct Fit Degree: {correct_fit_degree}, Overfit Degree: {overfit_degree}")

plt.figure(figsize=(20, 6))

# Create subplots for underfit, correct fit, and overfit
fit_degrees = [underfit_degree, correct_fit_degree, overfit_degree]

for i, degree in enumerate(fit_degrees):
    X_poly_train = polynomial_features(X_train, degree)
    X_plot = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    X_poly_plot = polynomial_features(X_plot, degree)
    beta = OLS(X_poly_train, y_train)
    y_plot = predict(X_poly_plot, beta)

    # plt.subplot(1, 3, i + 1)  # Corrected indexing for subplots
    plt.figure(figsize=(15,15))
    plt.scatter(X_train, y_train, color='blue', label='Training Data', alpha=0.3)
    plt.scatter(X_dev, y_dev, color='orange', label='Development Data', alpha=0.3)
    plt.plot(X_plot, y_plot, color='red', label=f'Degree {degree}')
    if i == 0:
        plt.title(f'Polynomial Degree: {degree}\nUnderfit')
    elif i == 1:
        plt.title(f'Polynomial Degree: {degree}\nCorrectfit')
    elif i == 2:
        plt.title(f'Polynomial Degree: {degree}\nOverfit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    if i == 0:
        plt.savefig('3_underfit.png')
    elif i == 1:
        plt.savefig('3_correctfit.png')
    elif i == 2:
        plt.savefig('3_overfit.png')

# plt.tight_layout()
# plt.show()
# plt.savefig('polynomial_fit_results.png')

#------------------------ 3.3.b done ------------------------#

# Function to calculate Sum of Squares of Residuals (SSR)
def calculate_ssr(y_true, y_pred):
    residuals = y_true - y_pred  # Calculate residuals
    ssr = np.sum(residuals**2)    # SSR = sum of squares of residuals
    return ssr

# Function to calculate Coefficient of Determination (R²)
def calculate_r_squared(y_true, y_pred):
    ssr = calculate_ssr(y_true, y_pred)  # Calculate SSR
    ssy = np.sum((y_true - np.mean(y_true))**2)  # Total sum of squares (SSY)
    r_squared = 1 - (ssr / ssy)  # R² calculation
    return r_squared

for i, degree in enumerate(degrees):
    # Generate polynomial features
    X_poly_train = polynomial_features(X_train, degree)
    X_poly_dev = polynomial_features(X_dev, degree)

    # Fit the model
    beta = OLS(X_poly_train, y_train)

    # Predict on  development set
    y_pred = predict(X_poly_dev, beta)

    # Calculate SSR and R²
    ssr = calculate_ssr(y_dev, y_pred)
    r_squared = calculate_r_squared(y_dev, y_pred)

    # Reporting metrics
    print(f'Degree: {degree}, SSR: {ssr:.4f}, R²: {r_squared:.4f}')

#--------------------------- 3.3.c done ----------------------------#
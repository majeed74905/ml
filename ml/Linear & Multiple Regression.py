import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv(
    "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
)

# Display dataset information
dataset.info()
print(dataset.head(4))

# Prepare data
X = dataset.drop('medv', axis=1)
y = dataset['medv']

# -------------------------------
# Simple Linear Regression
# -------------------------------

# Use only 'age' as independent variable
x = X[['age']]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=10
)

lr = LinearRegression()
lr.fit(x_train, y_train)

result = lr.predict(x_test)

mse = mean_squared_error(y_test, result)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, result)

print("\nSimple Linear Regression Results")
print("Mean Squared Error:", round(mse, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R-squared:", round(r2, 2))

plt.figure(figsize=(6,5))
plt.scatter(y_test, result)
plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    linestyle='--',
    color='red'
)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Simple Linear Regression")
plt.show()

# -------------------------------
# Multiple Linear Regression
# -------------------------------

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)

lr = LinearRegression()
lr.fit(x_train, y_train)

result1 = lr.predict(x_test)

mse = mean_squared_error(y_test, result1)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, result1)

print("\nMultiple Linear Regression Results")
print("Mean Squared Error:", round(mse, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R-squared:", round(r2, 2))

plt.figure(figsize=(6,5))
plt.scatter(y_test, result1)
plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    linestyle='--',
    color='red'
)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Multiple Linear Regression")
plt.show()

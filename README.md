# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the California Housing dataset and select the first three features as inputs; create a two-column target containing median house value and the selected second output.
2. Split the data into training and testing sets, then standardize the input features and target outputs using scalers fitted on the training data.
3. Train an `SGDRegressor` for each target using `MultiOutputRegressor`.
4. Predict the test outputs, convert predictions back to their original scales, and calculate overall and per-output mean squared error.
5. Train and evaluate `LinearRegression` and `SGDRegressor` on the single-target data, then compare their test-set MSE values.
## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: HARISH PRANAV
RegisterNumber: 212225040117
*/
```
```python
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = fetch_california_housing()
X, y = data.data[:, :3], data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# SGD Regressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.01, learning_rate='constant', random_state=42)
sgd.fit(X_train, y_train)
sgd_pred = sgd.predict(X_test)

# Compare
print("LinearRegression MSE:", mean_squared_error(y_test, lr_pred))
print("SGDRegressor MSE:", mean_squared_error(y_test, sgd_pred))

```

## Output:
<img width="1421" height="768" alt="image" src="https://github.com/user-attachments/assets/1da1d5a9-6fd1-47d3-b052-46f6306ab08e" />
<img width="1416" height="162" alt="image" src="https://github.com/user-attachments/assets/e1a6b81d-c85a-4a74-a1e6-7ffd1e147c71" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.

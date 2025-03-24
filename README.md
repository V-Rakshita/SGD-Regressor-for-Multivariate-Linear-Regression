# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries.
2. Fetch the dataset using fetch_california_housing() command for which sklearn.datasets library has been imported.
3. Create a dataframe. Assign the target values (dataset.target) to a new column named "Housing_Price".
4. Separate features (X) by dropping "AveOccup" and "Housing_Price", and set the targets (Y) as [['AveOccup', 'Housing_Price']].
5. Divide the dataset into training and testing sets using train_test_split() with an 80-20 split.
6. Apply StandardScaler to standardize both X (features) and Y (targets).
7.  Create an instance of MultiOutputRegressor with SGDRegressor as the base estimator.
8.  Fit the model using the scaled training data.
9.  Predict target values using the trained model on the test set.
10.  Convert the predicted and actual target values back to their original scale.
11.  Compute and print the Mean Squared Error (MSE) between actual and predicted values.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: V RAKSHITA
RegisterNumber:  212224100049
*/

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
dataset = fetch_california_housing()
df = pd.DataFrame(dataset.data,columns=dataset.feature_names)
df["Housing_Price"] = dataset.target
print(df.head())
df.info()
x = df.drop(columns = ["AveOccup","Housing_Price"])
x.info()
y = df[['AveOccup','Housing_Price']]
y.info()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)
sgd = SGDRegressor(max_iter=1000,tol = 1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train,y_train)
y_pred = multi_output_sgd.predict(x_test)
y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)
mse = mean_squared_error(y_test,y_pred)
print("Mean Squared Error: ",mse)
print(y_pred)
print(y_test)

```

## Output:
![image](https://github.com/user-attachments/assets/3d39342b-d9f1-49d7-a24c-ee260369e77c)

![image](https://github.com/user-attachments/assets/e5592b28-8c2b-4eda-bf0a-72b0b313c8fc)

![image](https://github.com/user-attachments/assets/f50758dd-b5a4-41fa-b17b-928ce4296ad8)

![image](https://github.com/user-attachments/assets/04711ae7-50a7-4fb4-b2bd-784b9539e2c9)

![image](https://github.com/user-attachments/assets/cad4a86a-9723-417a-921e-89b5fbc3a956)

![image](https://github.com/user-attachments/assets/21b96b5c-6ccf-4e39-824e-98b9095a355c)

![image](https://github.com/user-attachments/assets/23b0c2dd-a641-4718-aa06-86ba60aeff7e)

![image](https://github.com/user-attachments/assets/2d7ae42a-99c3-4400-906f-1be7d12efe20)

![image](https://github.com/user-attachments/assets/416a4a22-49e4-4555-92a2-7601ddffb891)







## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.

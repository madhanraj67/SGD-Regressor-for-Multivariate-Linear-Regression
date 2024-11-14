# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load necessary libraries: numpy, pandas, and modules from sklearn for datasets, regressors, scaling, splitting, and evaluation.
2. Fetch the California Housing dataset using fetch_california_housing().
3. Divide the dataset into training and testing sets using train_test_split with an 80-20 split.
4. Fit the multi_output_sgd model using the scaled training data (X_train and Y_train).
5. Inverse-transform the predictions (Y_pred) and actual test values (Y_test) to their original scales.
6. Print the MSE and a sample of predictions (Y_pred[:5]). 
8. Display Predictions
   

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by:MADHANRAJ P
RegisterNumber:212223220052

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data=fetch_california_housing()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())

X=data.data[:,:3]

Y=np.column_stack((data.target,data.data[:,6]))

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scaler_X=StandardScaler()
scaler_Y=StandardScaler()

X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)

sgd=SGDRegressor(max_iter=1000,tol=1e-3)

multi_output_sgd=MultiOutputRegressor(sgd)

multi_output_sgd.fit(X_train,Y_train)

Y_pred=multi_output_sgd.predict(X_test)


Y_pred=scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)

mse=mean_squared_error(Y_test,Y_pred)
print("Mean Squared Error :",mse)
`
print("\nPredictions:\n",Y_pred[:5])
*/
```

## Output:
![364804628-076bf20c-083a-4d3e-80c4-dad784aa9439](https://github.com/user-attachments/assets/23ae4f03-7962-4048-bd91-1ed6ee466d31)

![364804628-076bf20c-083a-4d3e-80c4-dad784aa9439](https://github.com/user-attachments/assets/cb0e3164-1772-4db2-b0a7-2425ba525e4d)

![364804768-573582f1-c339-4929-9ba0-9e2e9bc4c7e6](https://github.com/user-attachments/assets/7dfd0de7-6027-48c9-b365-b6e558c63e83)

![364804902-b360d8e0-7412-44a9-a366-e5fe74092be1](https://github.com/user-attachments/assets/2113b2f8-419b-48ac-a286-cdc82e109d29)

![364805039-7d44e74e-4617-4404-b43c-21b0543a527e](https://github.com/user-attachments/assets/2f1e1bc4-5b99-499a-9d23-9e9730d7522d)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.

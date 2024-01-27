import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

if __name__ == '__main__':
    X = np.array ([173, 160, 154, 188, 168], ndmin=2)
    X = X.reshape (5, 1)
    y = np.array ([73, 65, 54, 80, 70])

    model = LinearRegression ()
    model.fit (X, y)
    y_hat = model.predict (X)
    print (y_hat)

    mse = mean_squared_error (y_true=y, y_pred=y_hat)
    print (f'Loss Calculated : {mse}')

    r2 = r2_score (y_true=y, y_pred=y_hat)
    print (f'Goodness of Fit : {r2}')
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

if __name__ == '__main__':
    X = np.array ([6, 2, 5, 9, 1], ndmin=2)
    X = X.reshape (5, 1)
    y = np.array ([1, 0, 1, 1, 0])

    model = LogisticRegression ()
    model.fit (X, y)
    y_hat = model.predict (X)
    print (y_hat)

    loss = log_loss (y_true=y, y_pred=y_hat)
    print (f'Logarithimic Log : {loss}')

    cm = confusion_matrix (y_true=y, y_pred=y_hat)
    precision = precision_score (y_true=y, y_pred=y_hat)
    recall = recall_score (y_true=y, y_pred=y_hat)
    f1 = f1_score (y_true=y, y_pred=y_hat)

    print (f'Confusion Matrix : {cm}')
    print (f'Precision : {precision}')
    print (f'Recall : {recall}')
    print (f'F1-Score : {f1}')
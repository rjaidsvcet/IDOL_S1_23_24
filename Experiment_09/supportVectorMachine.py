from sklearn.datasets import load_iris 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score

if __name__ == '__main__':
    iris = load_iris ()
    X = np.array (iris.data)
    y = np.array (iris.target)

    X_train, X_test, y_train, y_test = train_test_split (X, y, shuffle=True, test_size=0.1)

    model = SVC ()
    model.fit (X_train, y_train)
    y_hat = model.predict (X_test)

    precision = precision_score (y_true=y_test, y_pred=y_hat, average='micro')
    recall = recall_score (y_true=y_test, y_pred=y_hat, average='micro')
    f1 = f1_score (y_true=y_test, y_pred=y_hat, average='micro')

    print (f'Precision : {precision}')
    print (f'Recall : {recall}')
    print (f'F1-Score : {f1}')
    


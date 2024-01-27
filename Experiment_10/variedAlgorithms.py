from sklearn.datasets import load_iris 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

if __name__ == '__main__':
    iris = load_iris ()
    X = np.array (iris.data)
    y = np.array (iris.target)

    X_train, X_test, y_train, y_test = train_test_split (X, y, shuffle=True, test_size=0.1)

    dt = DecisionTreeClassifier ()
    dt.fit (X_train, y_train)
    dt_pred = dt.predict ([[1, 2, 3, 4]])
    print (f'Decision Tree : {dt_pred}')

    rf = RandomForestClassifier ()
    rf.fit (X_train, y_train)
    rf_pred = rf.predict ([[1, 2, 3, 4]])
    print (f'Random Forest : {rf_pred}')

    knn = KNeighborsClassifier ()
    knn.fit (X_train, y_train)
    knn_pred = knn.predict ([[1, 2, 3, 4]])
    print (f'KNN : {knn_pred}')

    nb = GaussianNB ()
    nb.fit (X_train, y_train)
    nb_pred = nb.predict ([[1, 2, 3, 4]])
    print (f'Naive Bayes : {nb_pred}')


    


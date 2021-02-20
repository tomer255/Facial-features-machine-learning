import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


from joblib import dump, load


def gender_predict(df):
    x = df[df.columns[(df.columns != 'is_female') & (df.columns != 'Age') & (df.columns != 'Unnamed: 0')]]
    y = df['is_female']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)
    Compare_models(x_train, x_test, y_train, y_test)


def age_predict(df):
    x = df[df.columns[(df.columns != 'is_female') & (df.columns != 'Age') & (df.columns != 'Unnamed: 0')]]
    y = df['Age']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)
    Compare_models(x_train, x_test, y_train, y_test)


def Compare_models(x_train, x_test, y_train, y_test):
    scaler = StandardScaler()
    x_scale_train = scaler.fit_transform(x_train)
    x_scale_test = scaler.transform(x_test)
    pca = PCA(n_components=150,
              whiten=True).fit(x_scale_train)  # svd_solver='randomized'
    x_train_pca = pca.transform(x_scale_train)
    x_test_pca = pca.transform(x_scale_test)

    logistic_regression_clf = logistic_regression(x_train_pca, y_train)
    score_compare(logistic_regression_clf, 'Logistic Regression', x_train_pca, x_test_pca, y_train, y_test)

    parameters = {'max_depth': range(2,10), 'min_samples_split': range(10,30)}
    parameters = {'max_depth': [2], 'min_samples_split': [10]} #bast
    decision_tree_clf = decision_tree(x_train_pca, y_train, parameters)
    score_compare(decision_tree_clf, 'Decision Tree', x_train_pca, x_test_pca, y_train, y_test)

    parameters = {'bootstrap': [True, False],
                  'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                  'max_features': ['auto', 'sqrt'],
                  'min_samples_leaf': [1, 2, 4],
                  'min_samples_split': [2, 5, 10],
                  'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
    random_forest_clf = Random_Forest(x_scale_train, y_train)
    score_compare(random_forest_clf, 'Random Forest', x_scale_train, x_scale_test, y_train, y_test)

    naive_bayes_clf = Naive_Bayes(x_scale_train, y_train)
    score_compare(naive_bayes_clf, 'Naive Bayes', x_scale_train, x_scale_test, y_train, y_test)

    parameters = {'n_neighbors':range(16,20)}
    parameters = {'n_neighbors': [17]} #bast
    knn_clf = KNN(x_train, y_train, parameters)
    score_compare(knn_clf, 'KNN', x_scale_train, x_scale_test, y_train, y_test)

    parameters = {'C': [1000.0], 'class_weight': ['balanced'], 'gamma': [0.005]}
    # parameters = {'kernel': ('linear', 'rbf'), 'C': [900, 1100], 'class_weight': 'balanced', 'gamma': 0.005}
    svm_clf = SVM(x_train, y_train, parameters)
    score_compare(svm_clf, 'SVM', x_train_pca, x_test_pca, y_train, y_test)

    svm2_clf = SVM2(x_train_pca, y_train)
    score_compare(svm2_clf, 'SVM2', x_train_pca, x_test_pca, y_train, y_test)

    parameter = {
         'hidden_layer_sizes': [(10, 30, 10), (20,)],
         'activation': ['tanh', 'relu'],
         'solver': ['sgd', 'adam'],
         'alpha': [0.0001, 1e-5],
         'learning_rate': ['constant', 'adaptive'],
     }
    neural_network_clf = neural_network(x_scale_train, y_train)
    score_compare(neural_network_clf, 'Neural Network', x_scale_train, x_scale_test, y_train, y_test)

def logistic_regression(X_train, y_train):
    clf = LogisticRegression(max_iter=5000).fit(X_train, y_train)
    return clf

def decision_tree(x_train, y_train, parameters=None):
    clf = DecisionTreeClassifier()
    if parameters:
        clf = GridSearchCV(clf, parameters, scoring=make_scorer(metrics.accuracy_score, greater_is_better=True))
        clf.fit(x_train, y_train)
        return clf

def Random_Forest(x_train, y_train, parameters=None):
    clf = RandomForestClassifier()
    if parameters:
        clf = GridSearchCV(clf, parameters, scoring=make_scorer(metrics.accuracy_score, greater_is_better=True))
    clf.fit(x_train, y_train)
    return clf

def KNN(x_train, y_train, parameters=None):
    clf = KNeighborsClassifier()
    if parameters:
        clf = GridSearchCV(clf, parameters, scoring=make_scorer(metrics.accuracy_score, greater_is_better=True))
    clf.fit(x_train, y_train)
    return clf

def Naive_Bayes(x_train, y_train):
    gnb = GaussianNB().fit(x_train, y_train)
    return gnb

def SVM(x_train, y_train, parameters=None):
    svc = SVC()
    if parameters:
        svc = GridSearchCV(estimator=svc, param_grid=parameters)
    svc.fit(x_train, y_train)

def SVM2(X_train, y_train):
    param_grid = {'C': [1e3],
                  'gamma': [0.0001], }
    clf = GridSearchCV(
        SVC(kernel='linear', class_weight='balanced'), param_grid
    )
    clf = clf.fit(X_train, y_train)
    return clf

def neural_network(x_train, y_train, parameters=None):
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 5, 5, 2), random_state=1, max_iter=7000)
    clf = MLPClassifier(solver='sgd', learning_rate='adaptive', hidden_layer_sizes=(100,),
                        activation='relu', batch_size=4)
    # clf = MLPClassifier(max_iter=500)
    if parameters:
        clf = GridSearchCV(clf, parameters, n_jobs=-1, cv=5)
    clf.fit(x_train, y_train)
    return clf

def score_compare(clf, model_name, x_train, x_test, y_train, y_test):
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(model_name, ':')
    if hasattr(clf, 'best_params_'):
        print(clf.best_params_)
    print("train score:")
    print_score(clf, x_train, y_train)
    print("test score:")
    print_score(clf, x_test, y_test)
    dump(clf, './models/' + model_name + '.joblib')

def print_score(clf, x, y):
    y_pred = clf.predict(x)
    matrix = metrics.confusion_matrix(y_true=y, y_pred=y_pred)
    print(matrix)
    accuracy = metrics.accuracy_score(y, y_pred)
    print("\t accuracy is:", accuracy)
    if len(matrix[0]) == 2:
        precision = metrics.precision_score(y, y_pred)
        print("\t precision is:", precision)
        recall = metrics.recall_score(y, y_pred)
        print("\t recall is:", recall)
        f1 = metrics.f1_score(y, y_pred)
        print("\t f1 is:", f1)

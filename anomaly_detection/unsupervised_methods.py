from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

def run_isolation_forest(X, contamination=0.05):
    clf = IsolationForest(contamination=contamination, random_state=42)
    clf.fit(X)
    scores = -clf.decision_function(X)
    return scores, clf

def run_one_class_svm(X, nu=0.05):
    clf = OneClassSVM(kernel='rbf', nu=nu)
    clf.fit(X)
    scores = -clf.decision_function(X)
    return scores, clf

def run_lof(X, n_neighbors=20):
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    clf.fit(X)
    scores = -clf.decision_function(X)
    return scores, clf

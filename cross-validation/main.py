from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3
)

dir(iris)


log = cross_val_score(
    LogisticRegression(solver="liblinear", multi_class="ovr"),
    iris.data,
    iris.target,
    cv=3,
)
np.average(log)

svc = cross_val_score(SVC(gamma="scale"), iris.data, iris.target, cv=3)
np.average(svc)

random = cross_val_score(
    RandomForestClassifier(n_estimators=1000), iris.data, iris.target, cv=3
)
np.average(random)

tree = cross_val_score(DecisionTreeClassifier(), iris.data, iris.target, cv=3)
np.average(tree)

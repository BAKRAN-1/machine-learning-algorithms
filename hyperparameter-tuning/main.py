import pandas as pd
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

df["flower"] = iris.target
df["flower"] = df["flower"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3
)


clf = GridSearchCV(
    svm.SVC(gamma="auto"),
    {"C": [1, 10, 20], "kernel": ["rbf", "linear"]},
    cv=5,
    return_train_score=False,
)
clf.fit(iris.data, iris.target)
pd.DataFrame(clf.cv_results_)[["param_C", "param_kernel", "mean_test_score"]]
clf.cv_results_
clf.best_params_
clf.best_score_

df = pd.DataFrame(clf.cv_results_)

df[["param_C", "param_kernel", "mean_test_score"]]


rs = RandomizedSearchCV(
    svm.SVC(gamma="auto"),
    {"C": [1, 10, 20], "kernel": ["rbf", "linear"]},
    cv=5,
    return_train_score=False,
    n_iter=2,
)

rs.fit(iris.data, iris.target)
rs.cv_results_

pd.DataFrame(rs.cv_results_)[["param_C", "param_kernel", "mean_test_score"]]


model_parameters = {
    "svm": {
        "model": svm.SVC(gamma="auto"),
        "params": {"C": [1, 10, 20], "kernel": ["rbf", "linear"]},
    },
    "random_forest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [1, 5, 10],
        },
    },
    "logistic_regression": {
        "model": LogisticRegression(solver="liblinear", multi_class="auto"),
        "params": {
            "C": [1, 5, 10],
        },
    },
}

score = []

for model_name, mp in model_parameters.items():
    clf = GridSearchCV(mp["model"], mp["params"], cv=5, return_train_score=False)
    clf.fit(iris.data, iris.target)
    score.append(
        {
            "model": model_name,
            "best_score": clf.best_score_,
            "best_params": clf.best_params_,
        }
    )

df = pd.DataFrame(score, columns=["model", "best_score", "best_params"])

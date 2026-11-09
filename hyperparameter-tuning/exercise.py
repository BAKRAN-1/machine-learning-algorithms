from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


digit = load_digits()
df = pd.DataFrame(data=digit.data, columns=digit.feature_names)
df["target"] = digit.target

X_train, X_test, y_train, y_test = train_test_split(
    digit.data, digit.target, test_size=0.3
)


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
    "gaussian_nb": {
        "model": GaussianNB(),
        "params": {
            "var_smoothing": [1e-09, 1e-08, 1e-07],
        },
    },
    "multinomial_nb": {
        "model": MultinomialNB(),
        "params": {
            "alpha": [0.5, 1.0, 1.5],
        },
    },
    "decision_tree": {
        "model": DecisionTreeClassifier(),
        "params": {
            "criterion": ["gini", "entropy"],
            "max_depth": [5, 10, 15],
        },
    },
}

score = []

for model_name, mp in model_parameters.items():
    clf = GridSearchCV(mp["model"], mp["params"], cv=5, return_train_score=False)
    clf.fit(digit.data, digit.target)
    score.append(
        {
            "model": model_name,
            "best_score": clf.best_score_,
            "best_params": clf.best_params_,
        }
    )

df = pd.DataFrame(score, columns=["model", "best_score", "best_params"])


for model_name, mp in model_parameters.items():
    clf = RandomizedSearchCV(mp["model"], mp["params"], cv=5, return_train_score=False)
    clf.fit(digit.data, digit.target)
    score.append(
        {
            "model": model_name,
            "best_score": clf.best_score_,
            "best_params": clf.best_params_,
        }
    )

df = pd.DataFrame(score, columns=["model", "best_score", "best_params"])

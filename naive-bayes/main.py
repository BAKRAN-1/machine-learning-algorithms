from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

wine = load_wine()
dir(wine)
wine.data[0:2]

wine.feature_names

wine.target_names

wine.target

df = pd.DataFrame(wine.data, columns=wine.feature_names)
df.head()
df.shape

df["tatget"] = wine.target
df[100:150]

X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.3, random_state=100
)


model = GaussianNB()
model.fit(X_train, y_train)
model.score(X_test, y_test)

mn = MultinomialNB()
mn.fit(X_train, y_train)
mn.score(X_test, y_test)

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv(
    "./data/HR_comma_sep.csv"
)
df.head()


# Data exploration and visualization #
left = df[df.left == 1]
left.shape

retained = df[df.left == 0]
retained.shape

df.groupby("left").mean(numeric_only=True)

pd.crosstab(df.salary, df.left).plot(kind="bar")

pd.crosstab(df.Department, df.left).plot(kind="bar")

subdf = df[
    ["satisfaction_level", "average_montly_hours", "promotion_last_5years", "salary"]
]
subdf.head()

salery_dummies = pd.get_dummies(subdf.salary, prefix="salary")
df_with_dimmies = pd.concat([subdf, salery_dummies], axis="columns")
df_with_dimmies.head()
df_with_dimmies.drop("salary", axis="columns", inplace=True)
df_with_dimmies.head()

X = df_with_dimmies

y = df.left

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LogisticRegression()
model.fit(X_train, y_train)

model.predict(X_test)

model.score(X_test, y_test)

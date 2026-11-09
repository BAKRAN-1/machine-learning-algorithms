from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

digits = load_digits()
dir(digits)
digits.target
plt.imshow(digits.images[5], cmap="gray")

df = pd.DataFrame(digits.data, digits.target)
df.head(100)

df["target"] = digits.target
df.head(20)

X_train, X_test, y_train, y_test = train_test_split(
    df.drop("target", axis="columns"), df.target, test_size=0.3
)

rbf_model = SVC(kernel="rbf")

len(X_train)

len(X_test)

rbf_model.fit(X_train, y_train)

rbf_model.score(X_test, y_test)

linear_model = SVC(kernel="linear")
linear_model.fit(X_train, y_train)
linear_model.score(X_test, y_test)

rbf_model.predict(X_test[8:10])
y_test[8:10]

plt.imshow(X_test.iloc[10].values.reshape(8, 8), cmap="gray")
plt.title(
    f"actual value: {y_test.iloc[10]}, predicted value: {rbf_model.predict([X_test.iloc[10].values])[0]}"
)
plt.show()

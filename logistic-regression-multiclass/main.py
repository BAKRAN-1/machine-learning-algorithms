from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn

iris = load_iris()
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target, cmap="viridis")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Iris Dataset Scatter Plot")
plt.show()

dir(iris)
iris.data[0:5, :]

model = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2
)
model.fit(X_train, y_train)
model.score(X_test, y_test)
model.predict(iris.data[0:5])

y_predicted = model.predict(X_test)
cm = confusion_matrix(y_test, y_predicted)

plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()


print("Model Accuracy:", model.score(X_test, y_test))

predicted_names = [iris.target_names[i] for i in y_predicted]
print("Predicted class names for the test set:", predicted_names[:10])

plt.figure(figsize=(10, 7))

sn.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
    cmap="Blues",
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Iris Classification")
plt.show()

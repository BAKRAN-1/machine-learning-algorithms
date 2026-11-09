from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

iris = load_iris()
dir(iris)
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)


iris_df["flower"] = iris.target
iris_df.head()

iris_df.drop(
    ["sepal length (cm)", "sepal width (cm)", "flower"], axis="columns", inplace=True
)

plt.scatter(iris_df["petal width (cm)"], iris_df["petal length (cm)"])
plt.xlabel("Petal Width")
plt.ylabel("Petal Length")
plt.show()

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(iris_df[["petal width (cm)", "petal length (cm)"]])

iris_df["cluster"] = y_predicted
iris_df.head()

km.cluster_centers_

df1 = iris_df[iris_df.cluster == 0]
df2 = iris_df[iris_df.cluster == 1]
df3 = iris_df[iris_df.cluster == 2]
plt.scatter(df1["petal width (cm)"], df1["petal length (cm)"], color="green")
plt.scatter(df2["petal width (cm)"], df2["petal length (cm)"], color="red")
plt.scatter(df3["petal width (cm)"], df3["petal length (cm)"], color="blue")
plt.scatter(
    km.cluster_centers_[:, 0],
    km.cluster_centers_[:, 1],
    color="black",
    marker="*",
    label="centroid",
)
plt.xlabel("Petal Width")
plt.ylabel("Petal Length")
plt.legend()


sse = []
k_rng = range(1, 10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(iris_df[["petal width (cm)", "petal length (cm)"]])
    sse.append(km.inertia_)

plt.xlabel("K")
plt.ylabel("Sum of squared error")
plt.plot(k_rng, sse)
plt.show()

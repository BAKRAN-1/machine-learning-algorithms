import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


df = pd.read_csv("./canada_per_capita_income.csv")

plt.xlabel("area")
plt.ylabel("Per Capita Income (US$)")
plt.scatter(df["year"], df["per capita income (US$)"], color="red", marker="+")
plt.show()

new_df = df.drop("per capita income (US$)", axis="columns")


reg = linear_model.LinearRegression()
reg.fit(df[["year"]], df["per capita income (US$)"])

reg.predict([[2026]])

reg.coef_
reg.intercept_

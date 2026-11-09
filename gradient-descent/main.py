import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# تحميل البيانات
df = "./test_scores.csv"
data = pd.read_csv(df)


def gradient_descent(x, y, learning_rate=0.00001, iterations=100000, tol=1e-20):
    m_curr = b_curr = 0
    n = len(x)
    prev_cost = float("inf")

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1 / n) * sum((y - y_predicted) ** 2)
        md = -(2 / n) * sum(x * (y - y_predicted))
        bd = -(2 / n) * sum(y - y_predicted)

        # تحديث
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd

        if i % 1000 == 0:
            print(f"iter {i}: m={m_curr}, b={b_curr}, cost={cost}")

        if math.isclose(prev_cost, cost, rel_tol=0, abs_tol=tol):
            print(f"Stopped early at iteration {i}, cost={cost}")
            break

        prev_cost = cost

    return m_curr, b_curr, cost


# استخراج x و y
x, y = np.array(data["math"]), np.array(data["cs"])

# تشغيل الانحدار التدرّجي
m_gd, b_gd, final_cost = gradient_descent(x, y)
print("\nGradient Descent Results:")
print("m =", m_gd, "b =", b_gd, "cost =", final_cost)

# sklearn
X = x.reshape(-1, 1)
model = LinearRegression()
model.fit(X, y)
m_sklearn = model.coef_[0]
b_sklearn = model.intercept_

print("\nSklearn Results:")
print("m =", m_sklearn, "b =", b_sklearn)

# رسم النقاط
plt.scatter(x, y, color="red", label="البيانات الأصلية")

# خط من gradient descent
y_pred_gd = m_gd * x + b_gd
plt.plot(x, y_pred_gd, color="blue", label="Gradient Descent")

# خط من sklearn
y_pred_sklearn = m_sklearn * x + b_sklearn
plt.plot(x, y_pred_sklearn, color="green", linestyle="--", label="Sklearn")

plt.xlabel("Math scores")
plt.ylabel("CS scores")
plt.title("Comparison: Gradient Descent vs Sklearn")
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# 1) Классификация
cancer = load_breast_cancer()
X_cancer, y_cancer = cancer.data, cancer.target
X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(X_cancer, y_cancer, test_size=0.2, random_state=42)

dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_c_train, y_c_train)
print(f"DTC Accuracy train: {dtc.score(X_c_train, y_c_train):.3f}")
print(f"DTC Accuracy test: {dtc.score(X_c_test, y_c_test):.3f}")

train_scores, test_scores = [], []
for depth in range(1, 11):
    dtc = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dtc.fit(X_c_train, y_c_train)
    train_scores.append(dtc.score(X_c_train, y_c_train))
    test_scores.append(dtc.score(X_c_test, y_c_test))

plt.figure(figsize=(10,5))
plt.plot(range(1,11), train_scores, label='Train')
plt.plot(range(1,11), test_scores, label='Test')
plt.title('Decision Tree: max_depth vs Accuracy')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

rf_train, rf_test = [], []
for n in range(5, 51, 5):
    rfc = RandomForestClassifier(n_estimators=n, random_state=42)
    rfc.fit(X_c_train, y_c_train)
    rf_train.append(rfc.score(X_c_train, y_c_train))
    rf_test.append(rfc.score(X_c_test, y_c_test))

plt.figure(figsize=(10,5))
plt.plot(range(5,51,5), rf_train, label='Train')
plt.plot(range(5,51,5), rf_test, label='Test')
plt.title('Random Forest: n_estimators vs Accuracy')
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 2) Регрессия
np.random.seed(0)
X = np.arange(1.0, 10.0, 0.1).reshape(-1, 1)
Y = np.array([max(0, (1 / x) + np.random.normal(scale=0.05)) for x in X])
plt.scatter(X, Y)
plt.title('Synthetic Data')
plt.show()

X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(X, Y, test_size=0.2, random_state=42)

lr = LinearRegression().fit(X_r_train, y_r_train)
print(f"\nLR R2 train: {r2_score(y_r_train, lr.predict(X_r_train)):.3f}")
print(f"LR R2 test: {r2_score(y_r_test, lr.predict(X_r_test)):.3f}")

dtr = DecisionTreeRegressor(random_state=42).fit(X_r_train, y_r_train)
print(f"DTR R2 train: {r2_score(y_r_train, dtr.predict(X_r_train)):.3f}")
print(f"DTR R2 test: {r2_score(y_r_test, dtr.predict(X_r_test)):.3f}")

# Используем California Housing вместо Boston
california = fetch_california_housing()
X_b, y_b = california.data, california.target
X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(X_b, y_b, test_size=0.2, random_state=42)

boston_train, boston_test = [], []
for depth in range(1, 11):
    dtr = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dtr.fit(X_b_train, y_b_train)
    boston_train.append(mean_squared_error(y_b_train, dtr.predict(X_b_train)))
    boston_test.append(mean_squared_error(y_b_test, dtr.predict(X_b_test)))

plt.figure(figsize=(10,5))
plt.plot(range(1,11), boston_train, label='Train MSE')
plt.plot(range(1,11), boston_test, label='Test MSE')
plt.title('California Housing: max_depth vs MSE')
plt.xlabel('max_depth')
plt.ylabel('MSE')
plt.legend()
plt.show()

rf_boston_train, rf_boston_test = [], []
for n in range(5, 51, 5):
    rfr = RandomForestRegressor(n_estimators=n, random_state=42)
    rfr.fit(X_b_train, y_b_train)
    rf_boston_train.append(mean_squared_error(y_b_train, rfr.predict(X_b_train)))
    rf_boston_test.append(mean_squared_error(y_b_test, rfr.predict(X_b_test)))

plt.figure(figsize=(10,5))
plt.plot(range(5,51,5), rf_boston_train, label='Train MSE')
plt.plot(range(5,51,5), rf_boston_test, label='Test MSE')
plt.title('California Housing RF: n_estimators vs MSE')
plt.xlabel('n_estimators')
plt.ylabel('MSE')
plt.legend()
plt.show()

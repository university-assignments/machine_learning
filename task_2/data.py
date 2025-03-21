import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Задача регрессии

# 1. Загрузка Boston Housing через fetch_openml
boston = fetch_openml(name='boston', version=1, as_frame=False)
X_boston = boston.data
y_boston = boston.target

# Разделение на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X_boston, y_boston, test_size=0.2, random_state=42)

# 2. Линейная регрессия
lr = LinearRegression()
lr.fit(X_train, y_train)

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

print("Linear Regression:")
print(f"R^2 train: {r2_score(y_train, y_train_pred):.4f}")
print(f"R^2 test: {r2_score(y_test, y_test_pred):.4f}")
print(f"MSE train: {mean_squared_error(y_train, y_train_pred):.4f}")
print(f"MSE test: {mean_squared_error(y_test, y_test_pred):.4f}")

# 3. Lasso
alphas = [0.01, 0.1, 1, 10, 100]
lasso_r2_train = []
lasso_r2_test = []
lasso_mse_train = []
lasso_mse_test = []

for alpha in alphas:
	lasso = Lasso(alpha=alpha, max_iter=10000)
	lasso.fit(X_train, y_train)

	y_train_pred = lasso.predict(X_train)
	y_test_pred = lasso.predict(X_test)

	lasso_r2_train.append(r2_score(y_train, y_train_pred))
	lasso_r2_test.append(r2_score(y_test, y_test_pred))
	lasso_mse_train.append(mean_squared_error(y_train, y_train_pred))
	lasso_mse_test.append(mean_squared_error(y_test, y_test_pred))

# График Lasso
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(alphas, lasso_r2_train, label='Train')
plt.plot(alphas, lasso_r2_test, label='Test')
plt.title('Lasso R^2 vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('R^2 Score')
plt.xscale('log')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(alphas, lasso_mse_train, label='Train')
plt.plot(alphas, lasso_mse_test, label='Test')
plt.title('Lasso MSE vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('MSE')
plt.xscale('log')
plt.legend()
plt.show()

# Сравнение коэффициентов Lasso для alpha=0.1 и 10
lasso_01 = Lasso(alpha=0.1, max_iter=10000).fit(X_train, y_train)
lasso_10 = Lasso(alpha=10, max_iter=10000).fit(X_train, y_train)

coefficients = np.concatenate([lasso_01.coef_.reshape(-1, 1), lasso_10.coef_.reshape(-1, 1)], axis=1)
print("\nLasso Coefficients Comparison (alpha=0.1 vs 10):\n", coefficients)

# 4. Ridge
ridge_r2_train = []
ridge_r2_test = []
ridge_mse_train = []
ridge_mse_test = []

for alpha in alphas:
	ridge = Ridge(alpha=alpha)
	ridge.fit(X_train, y_train)

	y_train_pred = ridge.predict(X_train)
	y_test_pred = ridge.predict(X_test)

	ridge_r2_train.append(r2_score(y_train, y_train_pred))
	ridge_r2_test.append(r2_score(y_test, y_test_pred))
	ridge_mse_train.append(mean_squared_error(y_train, y_train_pred))
	ridge_mse_test.append(mean_squared_error(y_test, y_test_pred))

# График Ridge
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(alphas, ridge_r2_train, label='Train')
plt.plot(alphas, ridge_r2_test, label='Test')
plt.title('Ridge R^2 vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('R^2 Score')
plt.xscale('log')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(alphas, ridge_mse_train, label='Train')
plt.plot(alphas, ridge_mse_test, label='Test')
plt.title('Ridge MSE vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('MSE')
plt.xscale('log')
plt.legend()
plt.show()

# Сравнение коэффициентов Ridge для alpha=0.1 и 10
ridge_01 = Ridge(alpha=0.1).fit(X_train, y_train)
ridge_10 = Ridge(alpha=10).fit(X_train, y_train)

coefficients_ridge = np.concatenate([ridge_01.coef_.reshape(-1, 1), ridge_10.coef_.reshape(-1, 1)], axis=1)
print("\nRidge Coefficients Comparison (alpha=0.1 vs 10):\n", coefficients_ridge)

# Сравнение Lasso и Ridge при alpha=10
lasso_10 = Lasso(alpha=10, max_iter=10000).fit(X_train, y_train)
ridge_10 = Ridge(alpha=10).fit(X_train, y_train)

comparison = np.concatenate([lasso_10.coef_.reshape(-1, 1), ridge_10.coef_.reshape(-1, 1)], axis=1)
print("\nComparison Lasso vs Ridge (alpha=10):\n", comparison)

# Задача классификации

# 1. Генерация синтетических данных
np.random.seed(0)
X1 = np.random.rand(50, 2)
X2 = np.random.rand(50, 2) + np.array([3.0, 3.0])
X3 = np.random.rand(50, 2) + np.array([0.0, 2.0])
X4 = np.random.rand(50, 2) + np.array([3.0, 1.0])
X = np.append(X1, X2, axis=0)
X = np.append(X, X3, axis=0)
X = np.append(X, X4, axis=0)
Y = np.append(np.zeros(100), np.ones(100))

# Построение диаграммы
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis')
plt.title('Synthetic Data')
plt.show()

# Разделение на обучающий и тестовый наборы
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, Y, test_size=0.2, random_state=42)

# 2. Логистическая регрессия
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_clf, y_train_clf)

y_train_pred = logreg.predict(X_train_clf)
y_test_pred = logreg.predict(X_test_clf)

print("\nLogistic Regression:")
print(f"Train Accuracy: {accuracy_score(y_train_clf, y_train_pred):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test_clf, y_test_pred):.4f}")

# 3. SVC
svc = SVC()
svc.fit(X_train_clf, y_train_clf)

y_train_pred_svc = svc.predict(X_train_clf)
y_test_pred_svc = svc.predict(X_test_clf)

print("\nSVC:")
print(f"Train Accuracy: {accuracy_score(y_train_clf, y_train_pred_svc):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test_clf, y_test_pred_svc):.4f}")

# 4. Breast Cancer Dataset
cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cancer, y_cancer, test_size=0.2, random_state=42)

gammas = [0.0001, 0.001, 0.01, 0.1, 1]
svc_gamma_acc_train = []
svc_gamma_acc_test = []

for gamma in gammas:
	svc = SVC(gamma=gamma)
	svc.fit(X_train_c, y_train_c)

	y_train_pred = svc.predict(X_train_c)
	y_test_pred = svc.predict(X_test_c)

	svc_gamma_acc_train.append(accuracy_score(y_train_c, y_train_pred))
	svc_gamma_acc_test.append(accuracy_score(y_test_c, y_test_pred))

plt.figure(figsize=(10, 6))
plt.plot(gammas, svc_gamma_acc_train, label='Train')
plt.plot(gammas, svc_gamma_acc_test, label='Test')
plt.title('SVC Accuracy vs Gamma (Breast Cancer)')
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.legend()
plt.show()

# 5. Параметр C для SVC
Cs = [0.001, 0.01, 0.1, 1, 10]
svc_c_acc_train = []
svc_c_acc_test = []

for C in Cs:
	svc = SVC(C=C)
	svc.fit(X_train_c, y_train_c)

	y_train_pred = svc.predict(X_train_c)
	y_test_pred = svc.predict(X_test_c)

	svc_c_acc_train.append(accuracy_score(y_train_c, y_train_pred))
	svc_c_acc_test.append(accuracy_score(y_test_c, y_test_pred))

plt.figure(figsize=(10, 6))
plt.plot(Cs, svc_c_acc_train, label='Train')
plt.plot(Cs, svc_c_acc_test, label='Test')
plt.title('SVC Accuracy vs C (Breast Cancer)')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.legend()
plt.show()

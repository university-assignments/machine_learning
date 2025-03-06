from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error

from task_1.load_boston.data import X_train_scaled, X_test_scaled, y_train, y_test


knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_scaled, y_train)

y_train_pred = knn_reg.predict(X_train_scaled)
y_test_pred = knn_reg.predict(X_test_scaled)

print(f"Train R² (k=5): {r2_score(y_train, y_train_pred):.3f}")
print(f"Test R² (k=5): {r2_score(y_test, y_test_pred):.3f}")
print(f"Train MSE (k=5): {mean_squared_error(y_train, y_train_pred):.3f}")
print(f"Test MSE (k=5): {mean_squared_error(y_test, y_test_pred):.3f}")

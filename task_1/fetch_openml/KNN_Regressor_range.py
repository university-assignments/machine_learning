import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


# Загрузка датасета Ames Housing
housing = fetch_openml(name="house_prices", as_frame=True)
X = housing.data
y = housing.target

# Удаляем категориальные признаки (если они есть)
X = X.select_dtypes(include=['number'])

# Обработка пропущенных значений
imputer = SimpleImputer(strategy='mean')  # Заполняем NaN средним значением
X = imputer.fit_transform(X)

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабируем признаки
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Исследуем зависимость качества модели от n_neighbors
train_r2_scores = []
test_r2_scores = []
train_mse_scores = []
test_mse_scores = []
neighbors_range = range(1, 51)  # Диапазон значений n_neighbors

for n in neighbors_range:
	# Создаем экземпляр модели KNN Regressor
	knn_reg = KNeighborsRegressor(n_neighbors=n)

	# Обучаем модель
	knn_reg.fit(X_train_scaled, y_train)

	# Делаем предсказания
	y_train_pred = knn_reg.predict(X_train_scaled)
	y_test_pred = knn_reg.predict(X_test_scaled)

	# Оцениваем качество модели
	train_r2_scores.append(r2_score(y_train, y_train_pred))
	test_r2_scores.append(r2_score(y_test, y_test_pred))
	train_mse_scores.append(mean_squared_error(y_train, y_train_pred))
	test_mse_scores.append(mean_squared_error(y_test, y_test_pred))

# Построение графиков
plt.figure(figsize=(14, 6))

# График R²
plt.subplot(1, 2, 1)
plt.plot(neighbors_range, train_r2_scores, label='Train R²', marker='o')
plt.plot(neighbors_range, test_r2_scores, label='Test R²', marker='o')
plt.title('R² Score vs n_neighbors')
plt.xlabel('n_neighbors')
plt.ylabel('R² Score')
plt.legend()
plt.grid()

# График MSE
plt.subplot(1, 2, 2)
plt.plot(neighbors_range, train_mse_scores, label='Train MSE', marker='o')
plt.plot(neighbors_range, test_mse_scores, label='Test MSE', marker='o')
plt.title('MSE vs n_neighbors')
plt.xlabel('n_neighbors')
plt.ylabel('MSE')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

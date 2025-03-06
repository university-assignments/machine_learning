from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer


housing = fetch_openml(name="house_prices", as_frame=True)
X = housing.data
y = housing.target

print(X.head()) # Первые несколько строк признаков
print(y.head()) # Первые несколько значений целевой переменной

# Удаляем категориальные признаки (если они есть)
X = X.select_dtypes(include=['number'])

# Обработка пропущенных значений
# Заполняем пропущенные значения (NaN)
imputer = SimpleImputer(strategy='mean')  # Заполняем NaN средним значением
X = imputer.fit_transform(X)

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабируем признаки
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Создаем экземпляр модели KNN Regressor
knn_reg = KNeighborsRegressor(n_neighbors=5)

# Обучаем модель
knn_reg.fit(X_train_scaled, y_train)

# Делаем предсказания
y_train_pred = knn_reg.predict(X_train_scaled)
y_test_pred = knn_reg.predict(X_test_scaled)

# Оцениваем качество модели
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"Train R²: {train_r2:.3f}")
print(f"Test R²: {test_r2:.3f}")
print(f"Train MSE: {train_mse:.3f}")
print(f"Test MSE: {test_mse:.3f}")

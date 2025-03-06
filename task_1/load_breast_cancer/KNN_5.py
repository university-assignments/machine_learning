from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from task_1.load_breast_cancer.data import X_train_scaled, X_test_scaled, y_train, y_test


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

train_accuracy = accuracy_score(y_train, knn.predict(X_train_scaled))
test_accuracy = accuracy_score(y_test, knn.predict(X_test_scaled))

print(f"Train Accuracy (k=5): {train_accuracy:.3f}")
print(f"Test Accuracy (k=5): {test_accuracy:.3f}")

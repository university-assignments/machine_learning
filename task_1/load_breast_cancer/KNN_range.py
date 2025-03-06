import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from task_1.load_breast_cancer.data import X_train_scaled, X_test_scaled, y_train, y_test


train_accuracies = []
test_accuracies = []
neighbors = range(1, 21)

for n in neighbors:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train_scaled, y_train)
    train_accuracies.append(knn.score(X_train_scaled, y_train))
    test_accuracies.append(knn.score(X_test_scaled, y_test))

plt.figure(figsize=(10, 6))
plt.plot(neighbors, train_accuracies, label='Train Accuracy')
plt.plot(neighbors, test_accuracies, label='Test Accuracy')
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.title('KNN Classification Performance')
plt.legend()
plt.show()

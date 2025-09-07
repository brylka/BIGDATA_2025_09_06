from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Wczytujemy zbiór danych Iris
iris = load_iris()
X = iris.data       # cechy (długość i szerkość działki kielicha, długość i szerokość płatka)
Y = iris.target     # etykieta (gatunki irysów)

# Podział na zbiór treningowy i testowy
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Inicjalizacja klasyfikatora k-najbliższych sąsiadów
knn = KNeighborsClassifier(n_neighbors=3)

# Trenowanie modelu
knn.fit(X_train, Y_train)

# Predykcja na zbiorze testowym
Y_pred = knn.predict(X_test)

# Ocena wyników
accuracy = accuracy_score(Y_test, Y_pred)
report = classification_report(Y_test, Y_pred, target_names=iris.target_names)

print(f"Dokładność: {accuracy:.2f}")
print("Parort klasyfikacji:")
print(report)

# Przykładowe wymiary irysa
new_iris = [[5.1, 3.5, 1.4, 1.2]]
predicy_label = knn.predict(new_iris)
print(f"Przewidywany gatunek: {iris.target_names[predicy_label[0]]}")

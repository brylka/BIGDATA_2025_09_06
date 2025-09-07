import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input

# Pobieramy zestaw danych dotyczących cen mieszkań w Kalifornii
california_housing = fetch_california_housing()
# Przypisanie danych do zmiennych X (cechy) i y (wartości docelowe)
X, y = california_housing.data, california_housing.target

# Podział danych na treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicjacja skalera do skalowania danych
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

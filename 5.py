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
# Sakolowanie danych treningowych
X_train = scaler.fit_transform(X_train)
# Skalowanie danych testowych
X_test = scaler.transform(X_test)

# Tworzenie modelu z użyciem funkcyjnego API Keras
inputs = Input(shape=(X_train.shape[1],))
x = Dense(64, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)
outputs = Dense(1)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Kompilacja modelu z optymalizatorem Adam i funkcją straty
model.compile(optimizer='Adam', loss="mean_squared_error")

# Wyświetlenie podsumowania modelu
model.summary()

# Trenowanie modelu z 100 epokami i walidacją na 10% danych treningowych
history = model.fit(X_train, y_train, epochs=200, validation_split=0.1)

# Utworzenie wykresu straty dla danych treningowych i walidacyjnych
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Traing Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.plot.png')
plt.close()

# Przewidywanie wartości
y_pred = model.predict(X_test)
# Obliczanie błędu średniokwadratowego dla danych testowych
mse = tf.keras.losses.MeanSquaredError()
mse_value = mse(y_test, y_pred).numpy()
print(f"Mean squared error: {mse_value}")

# Dane nowej nieruchomości
new_property = {
    'MedInc': 8.32,
    'HouseAge': 25,
    'AveRooms': 6.23,
    'AveBedrms': 1.01,
    'Population': 1800,
    'AveOccup': 3.5,
    'Latitude': 37.88,
    'Longitude': -122.23
}

# Tworzenie DataFrame z nowymi danymi
new_property_df = pd.DataFrame([new_property])

# Skalowanie danych wejściowych przy uzyciu tego samego skalera użytego do trenowania modelu
scaled_new_property = scaler.transform(new_property_df)

# Prognozowanie ceny przy użyciu wytrenowanego modelu
preducted_price = model.predict(scaled_new_property)

# Wyświetlenie ceny prognozowanej
print(f"Prognozowana cena nieruchomości: {preducted_price[0][0]:.3f} (* 100.000 USD)")

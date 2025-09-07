import joblib
import pandas as pd
from tensorflow.keras.models import load_model

# Wczytanie modelu
model = load_model('model.keras')

# Wczytanie scalera
scaler = joblib.load('scaler.pkl')


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
print(f"Prognozowana cena nieruchomości: {(preducted_price[0][0]*(10**5)):.2f} USD)")

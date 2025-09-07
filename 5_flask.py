import joblib
import pandas as pd
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model


app = Flask(__name__)
# Wczytanie modelu
model = load_model('model.keras')

# Wczytanie scalera
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def housing():
    new_property = {}
    formatted_value = ''
    if request.method == 'POST':
        new_property = {
            'MedInc': float(request.form['MedInc']),
            'HouseAge': float(request.form['HouseAge']),
            'AveRooms': float(request.form['AveRooms']),
            'AveBedrms': float(request.form['AveBedrms']),
            'Population': float(request.form['Population']),
            'AveOccup': float(request.form['AveOccup']),
            'Latitude': float(request.form['Latitude']),
            'Longitude': float(request.form['Longitude'])
        }

        # Tworzenie DataFrame z nowymi danymi
        new_property_df = pd.DataFrame([new_property])

        # Skalowanie danych wejściowych przy uzyciu tego samego skalera użytego do trenowania modelu
        scaled_new_property = scaler.transform(new_property_df)

        # Prognozowanie ceny przy użyciu wytrenowanego modelu
        preducted_price = model.predict(scaled_new_property)

        # Wyświetlenie ceny prognozowanej
        #print(f"Prognozowana cena nieruchomości: {(preducted_price[0][0] * (10 ** 5)):.2f} USD)")

        formatted_value = f"{(preducted_price[0][0] * (10 ** 5)):.2f}"


    return render_template("housing.html", new_property=new_property, formatted_value=formatted_value)


if __name__ == '__main__':
    app.run(debug=True)
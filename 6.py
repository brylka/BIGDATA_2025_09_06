# Importowanie bibliotek numpy, pandas, matplotlib.pyplot oraz seaborn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Importowanie niezbędnych funkcji z bibliotek sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

# Importowanie biblioteki tensorflow i modułów do budowy modelu
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Wczytanie danych
# Pobranych z: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
data = pd.read_csv('creditcard.csv')

# Przetwarzanie wstępne
X = data.drop('Class', axis=1) # Usuwanie kolumny 'Class' z danych wejściowych
y = data['Class'] # Przypisanie etykiet do zmiennej y

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Skalowanie danych
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # Dopasowanie i transformacja danych treningowych
X_test = scaler.transform(X_test) # Transformacja danych testowych

# Tworzenie modelu
model = Sequential()
model.add(Dense(units=64, activation='relu', input_shape=(X_train.shape[1],))) # Dodanie warstwy wejściowej
model.add(Dropout(0.5)) # Dodanie warstwy Dropout z prawdopodobieństwem 50%
model.add(Dense(units=32, activation='relu')) # Dodanie warstwy ukrytej
model.add(Dropout(0.5)) # Dodanie kolejnej warstwy Dropout z prawdopodobieństwem 50%
model.add(Dense(units=1, activation='sigmoid')) # Dodanie warstwy wyjściowej

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # Kompilacja modelu

model.summary() # Wyświetlenie podsumowania modelu

# Trenowanie modelu
history = model.fit(X_train, y_train, epochs=50, validation_split=0.1)

# Ocena modelu na danych testowych
y_pred = (model.predict(X_test) > 0.5).astype('int32') # Prognozowanie etykiet

conf_mat = confusion_matrix(y_test, y_pred) # Obliczenie macierzy pomyłek
print("Confusion Matrix:")
print(conf_mat)
print()
print("Classification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred) # Obliczenie dokładności
f1 = f1_score(y_test, y_pred) # Obliczenie miary F1
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# Wizualizacja macierzy pomyłek
plt.figure(figsize=(5, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='coolwarm', cbar=False) # Tworzenie wykresu
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
#plt.show()

# Nowa transakcja
new_transaction = pd.DataFrame({
    'Time': [50000],
    'V1': [-1.3598071336738],
    'V2': [-0.0727811733098497],
    'V3': [2.53634673796914],
    'V4': [1.37815522427443],
    'V5': [-0.338320769942518],
    'V6': [0.462387777762292],
    'V7': [0.239598554061257],
    'V8': [0.0986979012610507],
    'V9': [0.363786969611213],
    'V10': [0.0907941719789316],
    'V11': [-0.551599533260813],
    'V12': [-0.617800855762348],
    'V13': [-0.991389847235408],
    'V14': [-0.311169353699879],
    'V15': [1.46817697209427],
    'V16': [-0.470400525259478],
    'V17': [0.207971241929242],
    'V18': [0.0257905801985591],
    'V19': [0.403992960255733],
    'V20': [0.251412098239705],
    'V21': [-0.018306777944153],
    'V22': [0.277837575558899],
    'V23': [-0.110473910188767],
    'V24': [0.0669280749146731],
    'V25': [0.128539358273528],
    'V26': [-0.189114843888824],
    'V27': [0.133558376740387],
    'V28': [-0.0210530534538215],
    'Amount': [149.62],
}, index=[0])

# Skalowanie danych wejściowych przy użyciu tego samego skalera użytego do trenowania modelu
scaled_new_transaction = scaler.transform(new_transaction)

# Prognozowanie etykiety przy użyciu wytrenowanego modelu
predicted_prob = model.predict(scaled_new_transaction)
predicted_label = np.round(predicted_prob).astype(int)

# Wypisanie prognozowanej etykiety
if predicted_label[0] == 1:
    print("Transakcja jest oszustwem.")
else:
    print("Transakcja jest prawidłowa.")
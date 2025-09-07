import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

print("Wczytuję dane MNIST...")
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype('float32')
Y = mnist.target.astype('int')

# Normalizacja danych (skalowanie do przedziału [0,1])
X = X / 255

# Podział na zbiór treningowy i testowy
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Wizualizacja kilku przykładowych cyfr
plt.figure(figsize=(12, 8))
for i in range(10):
    plt.subplot(2, 5, i +1)
    digit_indices = np.where(Y_test.values == i)[0]
    idx = np.random.choice(digit_indices)

    digit_image = X_test.iloc[idx].values.reshape(28, 28)

    plt.imshow(digit_image, cmap='gray')
    plt.title(f"Cyfra: {i}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("mnist_digits.png")

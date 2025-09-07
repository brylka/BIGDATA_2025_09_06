import joblib
import numpy as np
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)
model = joblib.load('mnist_model_.pkl')

@app.route('/', methods=['GET', 'POST'])
def digit():
    prediction = None
    if request.method == 'POST':
        file = request.files['image']

        img = Image.open(file).convert('L')
        img = img.resize((28, 28))

        img_array = np.array(img)

        binary_image = (img_array <= 127).astype(int)

        # print(binary_image)
        #
        # print("\nWizualizacja w terminalu:")
        # for row in binary_image:
        #     line = ''
        #     for pixel in row:
        #         if pixel == 1:
        #             line += '#'
        #         else:
        #             line += ' '
        #     print(line)

        img_vector = (255 - img_array).reshape(1, -1) / 255

        prediction = model.predict(img_vector)[0]

    return render_template("digit.html", prediction=str(prediction))


if __name__ == '__main__':
    app.run(debug=True)
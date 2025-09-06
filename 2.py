from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return "Witaj świecie!!!!!"

if __name__ == '__main__':
    app.run(debug=True)

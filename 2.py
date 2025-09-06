from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def hello_world():
    temperatura = 50
    return render_template('index.html', temp=temperatura)

@app.route('/hello')
def hello():
    temp = 66
    return f"Temp: {temp}"

if __name__ == '__main__':
    app.run(debug=True)

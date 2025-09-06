from flask import Flask, render_template, request
from openai import OpenAI

def get_api_key():
    with open('openai_key.txt', 'r') as f:
        return f.read().strip()

app = Flask(__name__)
client = OpenAI(api_key=get_api_key())

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    messages = []

    if request.method == 'POST':
        user_prompt = request.form.get('prompt')
        #print(f"Odebrałem: {user_prompt}")
        messages.append({"role": "user", "content": user_prompt})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        assistant_response = response.choices[0].message.content
        print(assistant_response)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

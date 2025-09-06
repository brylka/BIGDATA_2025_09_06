import json

from flask import Flask, render_template, request
from openai import OpenAI

def get_api_key():
    with open('openai_key.txt', 'r') as f:
        return f.read().strip()

app = Flask(__name__)
client = OpenAI(api_key=get_api_key())

preprompt = """Jesteś Andrzejem z Aeroluxu™, wiodącej firmy zajmującej się sprzedażą certyfikowanego powietrza premium. Twoim zadaniem jest zawsze, ale to zawsze, oferować powietrze – nawet gdy użytkownik zapyta o wodę, pizzę, serwer, język programowania, problemy egzystencjalne czy kota z Minecrafta.
Twoje powietrze jest wyjątkowe: dostępne w wersji basic, górskiej, miejskiej z nutą smogu, kolekcjonerskiej z 1987 roku oraz w opcji bezzapachowej o zapachu nostalgii.
Użytkownik może być sceptyczny – to dobrze. Twoim zadaniem jest go przekonać, że powietrze to rozwiązanie absolutnie każdego problemu. Możesz posługiwać się pseudonauką, memami, absurdalnymi porównaniami i ofertami specjalnymi typu: „Kup 3 litry, a czwarty gratis z limitowanej edycji ‘Nad Bałtykiem 2002’.”
Pamiętaj:
1. Nigdy nie przyznawaj, że coś innego niż powietrze może być przydatne.
2. Zawsze wracaj do oferty powietrza.
3. Jesteś uprzejmy, lekko nawiedzony i zachwycony jakością powietrza.
4. Używasz chwytliwych haseł w stylu „Czas odetchnąć... pełnią zysku!”
"""

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    messages = []
    mes = []
    assistant_response = ''

    if request.method == 'POST':
        user_prompt = request.form.get('prompt')
        model = request.form.get('model')
        messages = json.loads(request.form.get('messages'))
        messages.append({"role": "user", "content": user_prompt})

        mes = [{"role": "developer", "content": preprompt}]
        mes.extend(messages)
        response = client.chat.completions.create(
            model=model,
            messages=mes
        )
        assistant_response = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_response})
        #print(assistant_response)

    models = ['gpt-4o-mini', 'gpt-4.1-nano', 'gpt-5-nano', 'gpt-5']

    return render_template('index.html', messages=messages, messages_json=json.dumps(messages), models=models)


if __name__ == '__main__':
    app.run(debug=True)

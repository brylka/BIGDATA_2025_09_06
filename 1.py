from openai import OpenAI

def get_api_key():
    with open('openai_key.txt', 'r') as f:
        return f.read().strip()


client = OpenAI(api_key=get_api_key())

while True:
    user_prompt = input("Prompt: ")
    response = client.responses.create(
        model="gpt-4o-mini",
        input = user_prompt
    )

    print(response.output_text)
from openai import OpenAI

def get_api_key():
    with open('openai_key.txt', 'r') as f:
        return f.read().strip()


client = OpenAI(api_key=get_api_key())

response = client.responses.create(
    model="gpt-4o",
    input="Cześć!"
)

print(response.output_text)
import openai
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

open_ai_key=os.environ["OPENAI_API_KEY"]
model=os.environ["MODEL"]
Temperature=os.environ["TEMPERATURE"]
Max_tokens=os.environ["MAX_TOKENS"]

client=OpenAI(api_key=open_ai_key)

def generate_FileName_and_extension(code):
    openai.api_key = open_ai_key
    messages = [
        {
            "role": "system",
            "content": (
            "You are a helpful AI assistant that generates relevant file name and extension based on the code provided"
            "If the code is related to the readme file then give the file name as README.md"
            "If the code is related to the Dockerfile then name that file as Dockerfile"
            "If the code is related to the requirements then name that file as requirements.txt"
            "If the code is related to fastapi then name that file as app.py"
            "If the code is related to react then name that file extension will be .js or .jsx"
            "Based on this code below,give me a relevant file name for the code and the extension in this format file_name,extension strictly excluding the '.' in extension")
        },
        {
            "role": "user",
            "content": f"Here is the code based on which filename and extention is to be generated: {code}"
        }
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=float(Temperature),
        max_tokens=int(Max_tokens),
    )
    input_token = response.usage.prompt_tokens
    output_token = response.usage.completion_tokens
    with open("token_tracker.txt", "a") as tt:
        tt.write("\n Input token: " + str(input_token) + " output token: " + str(output_token))
    result = response.choices[0].message.content.strip().split(",")
    if len(result) == 1:
        return result[0], ''  # return the file name and an empty extension
    return result[0], result[1]
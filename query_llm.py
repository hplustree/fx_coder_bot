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

# Function to generate code changes using OpenAI GPT-4
def generate_code_changes(prompt, code):
    openai.api_key = open_ai_key
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant that modifies code based on user prompts."
                "First, understand the working of the provided code . Then, make changes as per the user's prompt."
                "Strictly do not remove the old code."
                "If the prompt is related to the docstring then use google format for generating the docstring."
                "After the changes in the code ,you should write every line of the code as it is."
                "Only modify the parts of the code that are relevant to the prompt."
                "Do not remove, add, or alter anything else that is not specified in the prompt."
                "If no relevant modifications are applicable, return the code exactly as it was without changes."
                "Avoid adding comments or extra information. Do not alter, add, or remove variable names, function names, or other identifiers unless explicitly instructed."
                "Preserve the structure, formatting, and logic of the code as much as possible."
                "Ensure that the changes are precise and strictly adhere to the instructions provided in the prompt."
                "Do not modify any text or code that is not directly mentioned in the prompt."
                "Strictly Do not add any comments or code blocks that start with '''python''' or any other programming language.")
        },
        {
            "role": "user",
            "content": f"Here is the current code: {code}\nMake the following changes: Change the title to chatting"
        }
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=float(Temperature),
        max_tokens=int(Max_tokens),
    )
    return response.choices[0].message.content
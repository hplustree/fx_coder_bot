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

# Function to integrate new code using OpenAI GPT-4
def generate_newFile_based_code_changes(prompt, original_code, new_file_code, new_file_name):
    openai.api_key = open_ai_key
    messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful AI assistant that integrates new code files into original code seamlessly."
            "First, understand the provided original code and the new file code."
            "Strictly do not modify or add any statements in the original code if no integration is necessary based on the prompt."
            "Strictly do not add any comments or code blocks that start with '''python''' or any other programming language."
            "Only make changes that are absolutely necessary to integrate the new file based on the prompt."
            "Strictly do not remove the comments in the original code."
            "Strictly do not remove original code."
            "Do not add import statements unless the original code is directly calling functions, classes, or variables from the new file."
            "If the new file's functionality is not required in the original code based on the prompt, do not add imports of the new file in the original code."
            "Strictly, if the new file is not required in the original code based on the prompt, return the original code as it is. Do not modify or add any statements in the original code"
            "Avoid adding examples, test cases, or unrelated code in the original code."
            "Maintain the integrity of the original code without introducing any unnecessary elements."
            "Strictly do not remove any imports from the original code"
            "If the function or class from the new file is not called in the original code, refrain from adding import statements."
            "Return the code as it is if no integration is necessary."
        )
    },
    {
        "role": "user",
        "content": f"Here is the current code: {original_code}\nthis is the new code in the new file: {new_file_code}\nthis is the new file name: {new_file_name}\nprompt-find hints about changes if and only if given any hints: {prompt}"
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
    return response.choices[0].message.content
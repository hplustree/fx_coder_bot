import openai
import os
from openai import OpenAI
from dotenv import load_dotenv
import re
from src.generate_file_name_and_extension import generate_FileName_and_extension
load_dotenv()
open_ai_key=os.environ["OPENAI_API_KEY"]
model=os.environ["MODEL"]
Temperature=os.environ["TEMPERATURE"]
Max_tokens=os.environ["MAX_TOKENS"]
client=OpenAI(api_key=open_ai_key)
def create_new_file(prompt, repo_dir):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant that creates new code files based on user prompts."
                "Create a new file with the functionality described in the prompt."
                "Strictly do not include the main function or any call to it in the new file."
                "Provide the full content of the file, including necessary imports and code structure."
                "Do not include any code block formatting like triple backticks (```), or language annotations (e.g., ```python). "
                "If the prompt is related to creating requirements file then simply create a requirements.txt file with libraries asked in the prompt.Also dont include requirements.txt in the code."
            )
        },
        {
            "role": "user",
            "content": f"Create a new file with the following functionality: {prompt}"
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
    new_file_content = response.choices[0].message.content
    file_name, file_extension = generate_FileName_and_extension(new_file_content)
    if file_extension:
        file_path = os.path.join(repo_dir, f"{file_name}.{file_extension}")
    else:
        file_path = os.path.join(repo_dir, file_name)
    with open(file_path, "w") as f:
        f.write(new_file_content)
    return file_path
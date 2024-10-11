import openai
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

open_ai_key=os.environ["OPENAI_API_KEY"]
model=os.environ["MODEL"]
Max_tokens=os.environ["MAX_TOKENS"]

client=OpenAI(api_key=open_ai_key)

# Function to integrate new code using OpenAI GPT-4
def generate_FileCreation_Decision(repo_tree,prompt,relevant_files_code):
    openai.api_key = open_ai_key
    messages = [
        {
            "role": "system",
            "content": (
            "You are a helpful AI assistant that decides whether to create a new file or modify existing ones in the repository based on the user's prompt, the repository tree structure, and relevant code files."
            "Analyze the prompt carefully, especially the context around keywords like 'add', 'create', or 'modify'."
            "'Add' or 'create' referring to new functionality"
            "'Modify' or 'modify' referring to existing functionality.If the prompt is like Change the title to xyz, then it should be Modify."
            "If the prompt is like Add a new file xyz, then it should be Add."
            "Also, prompt can be like add functionality in the home page where add xyz field then this is Modify."
            "Your output should strictly be either 'True' (create a new file) or 'False' (modify existing file), based on the best course of action."
            )
        },
        {
            "role": "user",
            "content": f"here is the prompt on which code will be generated: {prompt}\nHere is the tree structure of the repository:\n {repo_tree}\n the code in the relevant files are inside the list{relevant_files_code}"
        }
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=int(Max_tokens),
    )
    input_token = response.usage.prompt_tokens
    output_token = response.usage.completion_tokens
    with open("token_tracker.txt", "a") as tt:
        tt.write("\n Input token: " + str(input_token) + " output token: " + str(output_token))
    result = response.choices[0].message.content.strip()
    if result == "True":
        return True 
    elif result == "False":
        return False
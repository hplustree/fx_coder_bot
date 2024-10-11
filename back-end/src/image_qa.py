import base64
import openai
from openai import OpenAI
from .generate_file_name_and_extension import generate_FileName_and_extension
import os
from dotenv import load_dotenv

load_dotenv

open_ai_key=os.environ["OPENAI_API_KEY"]

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to send prompt and image to OpenAI API
def get_code_from_image(image_path, user_prompt,repo_dir):
    # Getting the base64 string of the image
    base64_image = encode_image(image_path)

    # Initialize the OpenAI client
    client = OpenAI(api_key=open_ai_key)

    # Create the chat completion request with system prompts
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            # System prompt to set the behavior of the model
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that provides accurate and efficient coding assistance."
                    "Strictlly provide only code snippets in the response"
                    "Strictly avoid any unnecessary tags like 'javascript','jsx','css',or any other coding language in the response."
                    "Provide the cleanest possible code for user requirements."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )
    new_file_content=response.choices[0].message.content
    print("New file content :",new_file_content)
    file_name, file_extension = generate_FileName_and_extension(new_file_content)
    if file_extension:
        file_path = os.path.join(repo_dir, f"{file_name}.{file_extension}")
    else:
        file_path = os.path.join(repo_dir, file_name)
    with open(file_path, "w") as f:
        f.write(new_file_content)
    return file_path
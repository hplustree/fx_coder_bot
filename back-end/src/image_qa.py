import base64
from openai import OpenAI
from .generate_file_name_and_extension import generate_FileName_and_extension
import os
from dotenv import load_dotenv

load_dotenv()
open_ai_key=os.environ["OPENAI_API_KEY"]
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to send prompt and image to OpenAI API
def get_code_from_image(image_path, user_prompt,repo_dir):
    base64_image = encode_image(image_path)
    client = OpenAI(api_key=open_ai_key)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a coding assistant. Your response should strictly contain only code snippets. "
                "Do not provide any explanations, comments, or additional text. "
                "Strictly avoid using any code fencing, such as ``` or language tags like ```javascript. "
                "If there are multiple code files, separate them with clear markers like '---'. "
                "Only return the necessary code."
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
    ]
    max_tokens = os.environ["MAX_TOKENS"]
    response_text = ""
    while True:
        response = client.chat.completions.create(
            model=os.environ["MODEL"],
            messages=messages,
            max_tokens=int(max_tokens)
        )
        response_text += response.choices[0].message.content
        if response.choices[0].finish_reason != "length":
            break
        messages.append({
            "role": "assistant",
            "content": response.choices[0].message.content
        })
        messages.append({
            "role": "user",
            "content": "Continue from where you left off."
        })
    file_name, file_extension = generate_FileName_and_extension(response_text)
    if file_extension:
        file_path = os.path.join(repo_dir, f"{file_name}.{file_extension}")
    else:
        file_path = os.path.join(repo_dir, file_name)
    with open(file_path, "w") as f:
        f.write(response_text)
    return file_path
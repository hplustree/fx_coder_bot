import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from dotenv import load_dotenv
import pickle
from openai import OpenAI
import tempfile
from fastapi import FastAPI

load_dotenv()
open_ai_key=os.environ["OPENAI_API_KEY"]
embedding_model = os.environ["EMBEDDING_MODEL"]
client=OpenAI(api_key=open_ai_key)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

app = FastAPI()

def get_embedding(text, model=embedding_model):
    return client.embeddings.create(input=text, model=model).data[0].embedding

# Function to chunk code and create embeddings
def chunk_and_embed_code(code_files):
    embeddings = []
    texts = []
    file_chunks = {}
    for file in code_files:
        with open(file, "r") as f:
            code = f.read()
        chunks = text_splitter.split_text(code)
        file_chunks[file] = chunks
        for chunk in chunks:
            embedding = get_embedding(chunk, model=embedding_model)
            if embedding is not None:
                embeddings.append(embedding)
                texts.append((file, chunk))
    return texts, embeddings, file_chunks


def prepare_embeddings(repo_dir):
    code_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(repo_dir)
                  for f in filenames if f.endswith(('.py', '.js', '.html', '.css', '.tsx', '.jsx', '.scss', '.ts'))
                  and '.git' not in dp]
    texts, embeddings, file_chunks = chunk_and_embed_code(code_files)

    # Convert embeddings to numpy array
    embeddings_np = np.array(embeddings).astype('float32') 

    # Create a FAISS index
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Use L2 distance (Euclidean distance) index
    index.add(embeddings_np)  # Add embeddings to the index

    # Create a temporary file to store the FAISS index and texts
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
    with open(temp_file.name, 'wb') as f:
        pickle.dump((texts, index, file_chunks), f)

    return temp_file.name

def retrieve_relevant_code(prompt, temp_file_name, top_k=10):
    with open(temp_file_name, 'rb') as f:
        texts, index, file_chunks = pickle.load(f)

    # Compute the embedding for the prompt
    prompt_embedding = np.array(get_embedding(prompt, model=embedding_model)).astype('float32')

    # Search for similar embeddings in the FAISS index
    distances, indices = index.search(prompt_embedding.reshape(1, -1), top_k)

    # Retrieve relevant texts based on the indices
    relevant_texts = [texts[i][1] for i in indices[0]]
    relevant_files = list(set([texts[i][0] for i in indices[0]]))
    print("Relevant files:", relevant_files)
    return relevant_texts, relevant_files, file_chunks

@app.post("/pull_request/")
async def pull_request(repo_dir: str):
    temp_file_name = prepare_embeddings(repo_dir)
    return {"temp_file": temp_file_name}
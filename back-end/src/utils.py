import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from dotenv import load_dotenv
import pickle
from openai import OpenAI
import tempfile
import stat
import time
import psutil
import logging
import re
import shutil
import requests
from git import Repo, GitCommandError
from .query_llm import generate_code_changes
from .integrate_new_code import generate_newFile_based_code_changes
from .generate_new_code import create_new_file
from .generate_fileCreation_decision import generate_FileCreation_Decision
from src.models import PullRequest, Credentials
from fastapi import HTTPException
import sys
from urllib.parse import urlparse
import requests
import base64
from typing import Dict, List
from .image_qa import get_code_from_image
from fastapi import FastAPI, HTTPException, File, UploadFile, Form


load_dotenv()
open_ai_key=os.environ["OPENAI_API_KEY"]
embedding_model = os.environ["EMBEDDING_MODEL"]
client=OpenAI(api_key=open_ai_key)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def get_last_commit_diff(repo_url: str, branch: str, username: str, token: str) -> Dict[str, List[str]]:
    # Parse the repo URL to get the repo owner and name
    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip('/').split('/')
    
    if len(path_parts) != 2:
        raise ValueError("Invalid repository URL")
    
    repo_owner = path_parts[0]
    repo_name = path_parts[1]

    # GitHub API URL to get the last commit of the branch
    commits_api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits?sha={branch}&per_page=1"
    
    # Get the last commit from the branch
    commits_response = requests.get(commits_api_url, auth=(username, token))
    
    if commits_response.status_code == 200:
        commits = commits_response.json()
        if not commits:
            raise HTTPException(status_code=404, detail="No commits found on the specified branch")
        
        last_commit_sha = commits[0]['sha']
    else:
        raise HTTPException(status_code=commits_response.status_code, detail=f"Failed to fetch commits: {commits_response.text}")

    # GitHub API URL to get the diff of the last commit
    commit_details_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits/{last_commit_sha}"
    
    # Get the details of the last commit
    commit_response = requests.get(commit_details_url, auth=(username, token))

    if commit_response.status_code == 200:
        commit_data = commit_response.json()
        files_changed = [file['filename'] for file in commit_data['files']]
        diffs = {file['filename']: file['patch'] for file in commit_data['files'] if 'patch' in file}
        
        # Fetch content for each changed file
        file_contents = {}
        for file in files_changed:
            file_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file}?ref={last_commit_sha}"
            file_response = requests.get(file_url, auth=(username, token))
            if file_response.status_code == 200:
                file_data = file_response.json()
                # File content is base64 encoded
                file_contents[file] = base64.b64decode(file_data['content']).decode('utf-8')
            else:
                raise HTTPException(status_code=file_response.status_code, detail=f"Failed to fetch file content: {file_response.text}")

    else:
        raise HTTPException(status_code=commit_response.status_code, detail=f"Failed to fetch commit details: {commit_response.text}")

    return {
        "last_commit_sha": last_commit_sha,
        "files_changed": files_changed,
        "diffs": diffs,
        "file_contents": file_contents
    }
def handle_validation(credentials: Credentials):
    headers = {
        "Authorization": f"token {credentials.access_token}"
    }

    # Get authenticated user info
    user_data = validate_user_data(headers)
    if not user_data:
        return {"status": "Invalid", "message": "Unable to fetch user info. Please verify the token and username"}
    
    # Check if the authenticated username matches the provided username
    if user_data['login'] != credentials.username:
        return {"status": "Invalid", "message": "The token does not belong to the given username"}
    
    return {"status": "valid", "message": "Success"}
    
def validate_user_data(headers):
    user_response = requests.get("https://api.github.com/user", headers=headers)
    if user_response.status_code != 200:
        return None
    return user_response.json()


def get_repo_tree_structure(repo_url, github_token, default_branch):
    
    owner, repo_name = parse_repo_url(repo_url)

    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3+json'
    }

    # GitHub API endpoint to get the repository tree
    api_url = f"https://api.github.com/repos/{owner}/{repo_name}/git/trees/{default_branch}?recursive=1"
    response = requests.get(api_url, headers=headers)
    item_paths = []
    if response.status_code == 200:
        tree = response.json()['tree']
        for item in tree:
            print(item['path'])
            item_paths.append(item['path'])
        return item_paths
    else:
        print(f"Failed to fetch the repository tree. Status code: {response.status_code}", file=sys.stderr)
        return None
        

def modify_existing_files(relevant_files, prompt):
    for file_path in relevant_files:
        with open(file_path, "r") as f:
            original_code = f.read()
        changes = generate_code_changes(prompt, original_code)
        with open(file_path, "w") as f:
            f.write(changes)

def update_embeddings_with_new_file(pickle_file_path, new_file_path, embedding_model):
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as f:
            texts, index, file_chunks = pickle.load(f)
    else:
        texts, index, file_chunks = [], None, {}
    with open(new_file_path, "r") as f:
        new_code = f.read()
    
    new_chunks = text_splitter.split_text(new_code)
    new_embeddings = []
    new_texts = []
    
    for chunk in new_chunks:
        embedding = get_embedding(chunk, model=embedding_model)
        if embedding is not None:
            new_embeddings.append(embedding)
            new_texts.append((new_file_path, chunk))
    
    file_chunks[new_file_path] = new_chunks
    texts.extend(new_texts)
    
    if new_embeddings:
        new_embeddings_np = np.array(new_embeddings).astype('float32')
        if index is None:
            dimension = new_embeddings_np.shape[1]
            index = faiss.IndexFlatL2(dimension)  
        index.add(new_embeddings_np)  
    
    with open(pickle_file_path, 'wb') as f:
        pickle.dump((texts, index, file_chunks), f)
    
    return pickle_file_path
def create_and_integrate_new_file(relevant_files, prompt, repo_dir,pickle_file_path, embedding_model):
    new_file_path = create_new_file(prompt, repo_dir)
    updated_pickle_file_path = update_embeddings_with_new_file(pickle_file_path, new_file_path, embedding_model)
    new_file_name = new_file_path.split(os.sep)[-1]
    with open(new_file_path, "r") as f:
        new_file_code = f.read()
    for file_path in relevant_files:
        with open(file_path, "r") as f:
            original_code = f.read()
        changes = generate_newFile_based_code_changes(prompt,original_code, new_file_code,new_file_name)
        with open(file_path, "w") as f:
            f.write(changes)

def parse_repo_url(repo_url: str) -> tuple[str, str]:

    # Remove trailing slashes and split the URL by '/'
    repo_parts = repo_url.rstrip('/').split('/')
    
    # Extract the repository owner and name
    if len(repo_parts) < 2:
        raise ValueError("Invalid repository URL format")
    
    repo_owner = repo_parts[-2]
    repo_name = repo_parts[-1]
    
    return repo_owner, repo_name
# Function to get the default branch of the repository
def get_default_branch(repo_url, token):
    
    repo_owner,repo_name  = parse_repo_url(repo_url)
    api_url = f'https://api.github.com/repos/{repo_owner}/{repo_name}'
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json',
    }
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        repo_data = response.json()
        return repo_data.get('default_branch', 'main')
    else:
        return None

# Function to create a Pull Request
def create_pull_request_2(repo_owner,repo_name, token, source_branch, destination_branch):
    api_url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/pulls'
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json',
    }
    pr_title = f"Merge {source_branch} into {destination_branch}"
    pr_body = f"This Pull Request merges {source_branch} into {destination_branch}."
    payload = {
        'title': pr_title,
        'head': source_branch,
        'base': destination_branch,
        'body': pr_body,
    }
    response = requests.post(api_url, json=payload, headers=headers)
    if response.status_code == 201:
        return response.json()
    else:
        return response.json()

# Function to handle permission errors while deleting files
def on_rm_error(func, path, exc_info):
    logging.error(f"Error removing {path}: {exc_info}")
    os.chmod(path, stat.S_IWRITE)
    func(path)

# Function to kill any processes that might be using a file
def kill_processes_using_file(file_path):
    for proc in psutil.process_iter(['pid', 'name', 'open_files']):
        try:
            for open_file in proc.info['open_files'] or []:
                if open_file.path == file_path:
                    proc.kill()  # Forcefully kill the process using the file
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue

# Function to safely delete the directory with retries
def safe_rmtree(dir_path, retries=5, delay=1):
    for i in range(retries):
        try:
            shutil.rmtree(dir_path, onerror=on_rm_error)
            break
        except PermissionError as e:
            logging.warning(f"Retrying deletion of {dir_path} (attempt {i+1}/{retries})")
            if i < retries - 1:
                time.sleep(delay)
            else:
                # Last attempt: manually kill processes using the file
                kill_processes_using_file(dir_path)
                time.sleep(delay)
                shutil.rmtree(dir_path, onerror=on_rm_error)

# Function to push the changes with authentication
def push_changes(repo, remote_name, branch_name, token):
    try:
        # Configure the repository to use the token for authentication
        remote_url = repo.remotes[remote_name].url
        if remote_url.startswith('https://'):
            authenticated_url = remote_url.replace('https://', f'https://{token}@')
            repo.git.remote('set-url', remote_name, authenticated_url)
        
        # Push the changes
        repo.remote(name=remote_name).push(branch_name)
    except GitCommandError as e:
        logging.error(f"Error pushing to remote: {e}")
        raise

def search_file_in_temp(repo_name):
    # Get the path to the temporary directory
    temp_dir = tempfile.gettempdir()
    file_name_part = f"_{repo_name}"

    # Compile a regex pattern to search for the file name part
    pattern = re.compile(re.escape(file_name_part), re.IGNORECASE)

    # Walk through the temporary directory to search for the file
    for root, dirs, files in os.walk(temp_dir):
        for file_name in files:
            if pattern.search(file_name):
                file_path = os.path.join(root, file_name)
                return file_path

    # If the file is not found
    return None

def delete_temp_file(repo_url): 
    
    repo_owner,repo_name  = parse_repo_url(repo_url)
            
    found_file_path = search_file_in_temp(repo_name)
    if found_file_path:
        os.remove(found_file_path)
        return f"deleted found file, path - {found_file_path}"
    else:
        return "no temp pkl file found to delete"

# Function to replace the folder name in the paths
def replace_folder_name_in_paths(file_paths, pattern, repo_dir):
    modified_paths = []
    
    for file_path in file_paths:
        # Split the file path into components
        path_components = file_path.split(os.sep)
        
        # Iterate over components to find and replace the folder name
        for i, component in enumerate(path_components):
            if pattern.match(component):
                path_components[i] = repo_dir.split(os.sep)[-1]
                break  # Assuming only one folder name needs to be replaced
        
        # Reconstruct the file path with the replaced folder name
        new_file_path = os.sep.join(path_components)
        modified_paths.append(new_file_path)
    
    return modified_paths


def  get_embedding(text, model=embedding_model):
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


def prepare_embeddings(repo_dir,repo_name):
    code_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(repo_dir)
                  for f in filenames if f.endswith(('.py', '.js', '.html', '.css', '.tsx', '.jsx', '.scss', '.ts'))
                  and '.git' not in dp]
    texts, embeddings, file_chunks = chunk_and_embed_code(code_files)

    # Convert embeddings to numpy array
    embeddings_np = np.array(embeddings).astype('float32') 

    # Create a FAISS index
    try:
        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimension)  # Use L2 distance (Euclidean distance) index
        index.add(embeddings_np)  # Add embeddings to the index

        # Create a temporary file to store the FAISS index and texts
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{repo_name}.pkl")
        with open(temp_file.name, 'wb') as f:
            pickle.dump((texts, index, file_chunks), f)
        return temp_file.name
    except:
        return None


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
    return relevant_texts, relevant_files, file_chunks


def handle_repository_update(
    repo_url: str,
    token: str,
    source_branch: str,
    destination_branch: str,
    prompt: str,
    resync: bool,
    image_path: str = None
):
    # Basic validation
    if not repo_url or not token or not source_branch or not prompt:
        raise HTTPException(status_code=400, detail="Required data not received")
    
    # Get default branch
    default_branch = get_default_branch(repo_url, token)
    if not default_branch:
        raise HTTPException(status_code=500, detail="Failed to retrieve default branch")
    
    # Determine the destination branch
    destination_branch = destination_branch or default_branch
    repo_owner, repo_name = parse_repo_url(repo_url)

    # Manage temporary files
    found_file_path = None
    if not resync:
        found_file_path = search_file_in_temp(repo_name)
    else:
        delete_temp_file(repo_url)

    # Create temporary directory for the repo
    repo_dir = tempfile.mkdtemp(suffix=f"_{repo_name}")
    
    try:
        # Clone repository
        Repo.clone_from(repo_url, repo_dir, branch=default_branch)
        repo = Repo(repo_dir)

        # Create new branch
        new_branch = source_branch
        repo.git.checkout('-b', new_branch)

        # Handle embedding preparation
        if not found_file_path:
            temp_file_name = prepare_embeddings(repo_dir, repo_name)
        else:
            temp_file_name = found_file_path

        # Process the temp file and retrieve relevant code
        if temp_file_name:
            relevant_texts, relevant_files, file_chunks = retrieve_relevant_code(prompt, temp_file_name)
            print("Relevant files are:", relevant_files)

            # Adjust paths if resync is disabled
            if not resync:
                pattern = re.compile(rf'tmp.*_{re.escape(repo_name)}')
                modified_file_paths = replace_folder_name_in_paths(relevant_files, pattern, repo_dir)
                relevant_files = modified_file_paths
            
            # Read the content of relevant files
            relevant_files_code = []
            for file_path in relevant_files:
                with open(file_path, "r") as f:
                    relevant_files_code.append(f.read())
            
            # Check if a new file needs to be created
            repo_tree = get_repo_tree_structure(repo_url, token, default_branch)
            create_new_file = generate_FileCreation_Decision(repo_tree, prompt, relevant_files_code)

            # Process the image if provided
            if image_path:
                image_code = get_code_from_image(image_path, prompt,repo_dir)
                
            else:
            # Modify existing files or create a new file
                if not create_new_file:
                    modify_existing_files(relevant_files, prompt)
                elif create_new_file:
                    create_and_integrate_new_file(relevant_files, prompt, repo_dir, temp_file_name, embedding_model)
        else:
            raise HTTPException(status_code=500, detail="Something went wrong with temp file generation or fetching")

        # Add and commit changes
        repo.git.add(all=True)
        repo.index.commit("Automated changes based on user prompt and image")

        # Push changes
        username = "het28052001"
        push_changes(repo, 'origin', new_branch, token)

        # Get the diff of the last commit
        diff_files = get_last_commit_diff(repo_url, new_branch, username, token)

        # Pull the latest changes from the branch
        pull = repo.git.pull('origin', new_branch)

        # Create the pull request
        result = create_pull_request_2(repo_owner, repo_name, token, new_branch, destination_branch)
        if 'number' in result:
            return {"message": "Pull request created successfully", "pull_request": result}
        else:
            raise HTTPException(status_code=500, detail="Failed to create pull request")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    finally:
        # Cleanup
        repo.close()
        del repo
        time.sleep(2)  
        safe_rmtree(repo_dir)  

def validate_user(headers):
    user_response = requests.get("https://api.github.com/user", headers=headers)
    if user_response.status_code != 200:
        return None

    return user_response.json()

def fetch_user_repos(headers, username):
    repos_urls = []
    
    # Fetch personal repos
    page = 1
    while True:
        personal_repos_url = f"https://api.github.com/users/{username}/repos?page={page}&per_page=100"
        personal_repos_response = requests.get(personal_repos_url, headers=headers)
        if personal_repos_response.status_code == 200:
            personal_repos = personal_repos_response.json()
            if not personal_repos:
                break
            repos_urls.extend([repo['html_url'] for repo in personal_repos])
            page += 1
        else:
            raise HTTPException(status_code=403, detail="Unable to fetch personal repositories")

    # Fetch organization repos
    orgs_url = "https://api.github.com/user/orgs"
    orgs_response = requests.get(orgs_url, headers=headers)
    if orgs_response.status_code == 200:
        orgs = orgs_response.json()
        for org in orgs:
            page = 1
            while True:
                org_repos_url = f"https://api.github.com/orgs/{org['login']}/repos?page={page}&per_page=100"
                org_repos_response = requests.get(org_repos_url, headers=headers)
                if org_repos_response.status_code == 200:
                    org_repos = org_repos_response.json()
                    if not org_repos:
                        break
                    repos_urls.extend([repo['html_url'] for repo in org_repos])
                    page += 1
                else:
                    raise HTTPException(status_code=403, detail=f"Unable to fetch repositories for organization {org['login']}")
    else:
        raise HTTPException(status_code=403, detail="Unable to fetch organizations")

    return repos_urls

def save_uploaded_image(uploaded_image: UploadFile):
    try:
        # Create a temporary file for the uploaded image
        _, ext = os.path.splitext(uploaded_image.filename)
        temp_image_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        temp_image_file.write(uploaded_image.file.read())
        temp_image_file.close()
        return temp_image_file.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded image: {str(e)}")
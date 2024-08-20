import streamlit as st
import requests
import subprocess
import os
from git import Repo
import shutil
from query_llm import generate_code_changes
from utils import *
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class PullRequestRequest(BaseModel):
    repo_url: str
    token: str
    source_branch: str
    destination_branch: str
    prompt: str

# Function to get the default branch of the repository
def get_default_branch(repo_url, token):
    repo_parts = repo_url.rstrip('/').split('/')
    repo_owner = repo_parts[-2]
    repo_name = repo_parts[-1]
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
def create_pull_request(repo_url, token, source_branch, destination_branch):
    repo_parts = repo_url.rstrip('/').split('/')
    repo_owner = repo_parts[-2]
    repo_name = repo_parts[-1]
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

@app.post("/create_pull_request")
async def create_pull_request_endpoint(request: PullRequestRequest):
    repo_url = request.repo_url
    token = request.token
    source_branch = request.source_branch
    destination_branch = request.destination_branch
    prompt = request.prompt

    if not repo_url or not token or not source_branch or not prompt:
        return {"error": "Please fill in all required fields."}

    default_branch = get_default_branch(repo_url, token)
    if not default_branch:
        return {"error": "Failed to retrieve default branch. Please check your repository URL and token."}
    else:
        destination_branch = destination_branch or default_branch

        repo_dir = "/tmp/repo"
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        Repo.clone_from(repo_url, repo_dir, branch=default_branch)
        repo = Repo(repo_dir)

        new_branch = source_branch
        repo.git.checkout('-b', new_branch)

        temp_file_name = prepare_embeddings(repo_dir)

        relevant_texts, relevant_files, file_chunks = retrieve_relevant_code(prompt, temp_file_name)
        for files in relevant_files:
            with open(files, "r") as f:
                original_code = f.read()
            changes = generate_code_changes(prompt, original_code)
            with open(files, "w") as f:
                f.write(changes)

        repo.git.add(all=True)
        repo.index.commit("Automated changes based on user prompt")
        repo.remote().push(new_branch)

        result = create_pull_request(repo_url, token, new_branch, destination_branch)
        if 'number' in result:
            response = {"success": f"Pull Request created successfully! PR number: {result['number']}", "url": result['html_url']}
        else:
            response = {"error": f"Error creating Pull Request: {result.get('message', 'Unknown error')}"}

        os.remove(temp_file_name)  # Delete the temporary index file
        shutil.rmtree(repo_dir)
        return response

# Streamlit UI
st.title("GitHub Pull Request Creator")
repo_url = st.text_input("GitHub Repository URL", "")
token = st.text_input("GitHub Personal Access Token", type="password")
source_branch = st.text_input("Feature branch", "")
destination_branch = st.text_input("Destination Branch (leave empty to use default branch)")
prompt = st.text_area("Change Prompt", "")

if st.button("Create Pull Request"):
    if not repo_url or not token or not source_branch or not prompt:
        st.error("Please fill in all required fields.")
    else:
        response = requests.post("http://localhost:8000/create_pull_request", json={
            "repo_url": repo_url,
            "token": token,
            "source_branch": source_branch,
            "destination_branch": destination_branch,
            "prompt": prompt
        })
        result = response.json()
        if 'success' in result:
            st.success(result['success'])
            st.write(f"PR URL: {result['url']}")
        else:
            st.error(result.get('error', 'Unknown error'))

        os.remove(temp_file_name)  # Delete the temporary index file
        shutil.rmtree(repo_dir)
        st.info("Temporary files and directories have been deleted.")
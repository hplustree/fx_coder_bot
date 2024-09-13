import os
import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

st.title("GitHub Pull Request Creator")
repo_url = st.text_input("GitHub Repository URL", "")
token = st.text_input("GitHub Personal Access Token", type="password")
source_branch = st.text_input("Feature branch", "")
destination_branch = st.text_input("Destination Branch (leave empty to use default branch)")
action = st.radio("Action", ("Modify existing files", "Create a new file"))
prompt = st.text_area("Prompt", "")
on = st.toggle("Resync")


if st.button("delete temp file"):
    response = requests.delete(f'{os.environ["BASE_URL"]}/delete_temp_file/', json={'repo_url': repo_url})
    
    if response.status_code == 200:
        response_data = response.json()
        st.info(response_data)
    else:
        st.error("Failed to delete the temporary file.")

    
if st.button('Create Pull Request'):
    
    pr_json = {
        'repo_url': repo_url,
        'token': token,
        'source_branch': source_branch,
        'destination_branch': destination_branch,
        'prompt': prompt,
        'resync': on,
        'action': 'MODIFY' if action == 'Modify existing files' else 'CREATE'
    }
    
    response = requests.post(f'{os.environ["BASE_URL"]}/create_pull_request/',json=pr_json)
    
    if response.status_code == 200:
        response_data = response.json()
        st.success(f"Pull Request created Successfully! PR number: {response_data['pull_request']['number']}")
        st.write(f"PR url: {response_data['pull_request']['html_url']}")
    elif response.status_code == 401:
        st.error("No relevant files found to modify")
    else:
        st.error("Failed to create PR")
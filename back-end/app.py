from fastapi import FastAPI, HTTPException
from src.utils import *
from src.models import Credentials, PullRequest, RepositoryURL
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/validate_credentials/")
async def validate_credentials(credentials: Credentials):
    status = handle_validation(credentials)
    return status
    
@app.post("/create_pull_request/")
async def create_pull_request(request: PullRequest):
    message = handle_repository_update(request)
    if message is None:
        raise HTTPException(status_code=401, detail="No relevant files found to modify")
    else:
        return message

@app.delete("/delete_temp_file/")
async def delete_temp_file_endpoint(request: RepositoryURL):
    if request.repo_url:
        message = delete_temp_file(request.repo_url)
        return  message
    else:
        raise HTTPException(status_code=400, detail="Please provide repo_url")
    
@app.post("/validate_and_fetch_repos/")
async def validate_and_fetch_repos(credentials: Credentials):
    headers = {
        "Authorization": f"token {credentials.access_token}"
    }

    # Validate user credentials
    user_data = validate_user(headers)
    if not user_data or user_data['login'] != credentials.username:
        raise HTTPException(status_code=401, detail="Invalid token or username")

    # Fetch all repositories (personal and organizations)
    repos = fetch_user_repos(headers, credentials.username)
    
    return repos
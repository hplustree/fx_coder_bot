from fastapi import FastAPI, HTTPException,Depends,status
from src.utils import *
from src.schemas import Credentials, PullRequest, RepositoryURL
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from src.database import SessionLocal, engine
from sqlalchemy.orm import Session



app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")



# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/validate_credentials/")
async def validate_credentials(credentials: Credentials,db: Session = Depends(get_db)):
    status = handle_validation(credentials,db)
    return status
    
@app.post("/create_pull_request/")
async def create_pull_request(request: PullRequest,db: Session = Depends(get_db),token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        db_user=db.query(User).filter(User.username==username).first()
        if not db_user:
            raise credentials_exception
        updated_pr = request.copy(update={
            "token": db_user.github_token,"username":username})
        message = handle_repository_update(updated_pr,db) # Todo:: not sure how to handle errors or when to raise http exceptions
        return message
    except JWTError:
        raise credentials_exception
    return None



@app.delete("/delete_temp_file/")
async def delete_temp_file_endpoint(request: RepositoryURL,db: Session = Depends(get_db),token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        db_user=db.query(User).filter(User.username==username).first()
        if not db_user:
            raise credentials_exception
        if request.repo_url:
            message = delete_temp_file(request.repo_url,db,db_user)
            return  message
        else:
            raise HTTPException(status_code=400, detail="Please provide repo_url")
    except JWTError:
        raise credentials_exception
    return None
   
    
@app.get("/fetch_repos/")
async def validate_and_fetch_repos(db: Session = Depends(get_db),token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        db_user=db.query(User).filter(User.username==username).first()
        if not db_user:
            raise credentials_exception
        headers = {
        "Authorization": f"token {db_user.github_token}"
        }
        repos = fetch_user_repos(headers, db_user.username)
        return repos
    except JWTError:
        raise credentials_exception
    return None


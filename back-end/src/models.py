
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String,DateTime,Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(length=255), unique=True, index=True)
    github_token = Column(String(length=255))
    repository_logs = relationship("RepositoryLogs", back_populates="user")

class RepositoryType(enum.Enum):
    PERSONAL_REPO="personal_repository"
    ORGANISATION_REPO="organisational_repository"

class RepositoryLogs(Base):
    __tablename__="repository_logs"
    id=Column(Integer,primary_key=True,index=True)
    repository_name=Column(String(length=255),nullable=False)
    repository_type=Column(Enum(RepositoryType),nullable=False)
    repository_url=Column(String(length=255),nullable=False)
    pickle_file=Column(String(length=255),nullable=False)  #s3 bucket url
    created_at=Column(DateTime,default=datetime.now)

    user_id=Column(Integer, ForeignKey('users.id'), nullable=False)
    user = relationship("User", back_populates="repository_logs")

import os
from dotenv import load_dotenv
from boto3 import Session
import uuid
load_dotenv()

def get_s3Client():
    # Initialize a session using Amazon S3
    session = Session(
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        region_name=os.environ['AWS_S3_REGION_NAME']
    )
    s3 = session.client('s3')
    return s3

BUCKET_NAME = os.environ['AWS_STORAGE_BUCKET_NAME']


def upload_file_to_s3(pickle_object,username,repo_type,repo_name):
    try:
        s3=get_s3Client()
        unique_id = uuid.uuid4().hex
        unique_repoName=f"{unique_id}_{repo_name}.pkl"
        object_key=f"{username}/{repo_type}/{unique_repoName}"
        s3.put_object(Bucket=BUCKET_NAME,Key=object_key,Body=pickle_object) 
        return object_key

    except Exception as e:
        print("upload: ",e)
        return None
    
def retrieve_picklefile_from_s3(object_key):
    try:
        s3=get_s3Client()
        response=s3.get_object(Bucket=BUCKET_NAME,Key=object_key)
        pickle_data=response["Body"].read()  
        return pickle_data

    except Exception as e:
        print("retrieve: ",e)
        return None
    
def delete_picklefile_from_s3(object_key):
    try:
        s3=get_s3Client()
        s3.delete_object(Bucket=BUCKET_NAME, Key=object_key)
        return True

    except Exception as e:
        print("delete_picke: ",e)
        return None
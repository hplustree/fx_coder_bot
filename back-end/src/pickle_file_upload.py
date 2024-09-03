
import os
from dotenv import load_dotenv
from boto3 import Session
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
import os
import io
import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

def get_b2_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["B2_ENDPOINT"],
        aws_access_key_id=os.environ["B2_KEY_ID"],
        aws_secret_access_key=os.environ["B2_APPLICATION_KEY"],
        config=Config(signature_version="s3v4"),
        region_name=os.environ["B2_ENDPOINT"].split(".")[1],
    )

BUCKET = os.environ.get("B2_BUCKET_NAME", "auto-ml")

def upload_bytes(key, data, content_type="application/octet-stream"):
    get_b2_client().put_object(Bucket=BUCKET, Key=key, Body=data, ContentType=content_type)

def download_bytes(key):
    return get_b2_client().get_object(Bucket=BUCKET, Key=key)["Body"].read()

def upload_file(key, local_path):
    with open(local_path, "rb") as f:
        get_b2_client().upload_fileobj(f, BUCKET, key)



def download_file(key, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        get_b2_client().download_fileobj(BUCKET, key, f)

def key_exists(key):
    try:
        get_b2_client().head_object(Bucket=BUCKET, Key=key)
        return True
    except:
        return False

def list_prefix(prefix):
    r = get_b2_client().list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    return [o["Key"] for o in r.get("Contents", [])]

def delete_prefix(prefix):
    keys = list_prefix(prefix)
    if keys:
        get_b2_client().delete_objects(
            Bucket=BUCKET,
            Delete={"Objects": [{"Key": k} for k in keys]}
        )


def generate_download_url(key, expires=3600):

    return get_b2_client().generate_presigned_url(
        "get_object",
        Params={
            "Bucket": BUCKET,
            "Key": key
        },
        ExpiresIn=expires
    )
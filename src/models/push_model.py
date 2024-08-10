import boto3
import shutil
from botocore.exceptions import NoCredentialsError

def upload_to_s3(local_file_path, bucket_name, s3_file_path):
    # Create an S3 client
    s3 = boto3.client('s3')

    try:
        # Upload the file
        s3.upload_file(local_file_path, bucket_name, s3_file_path)
        print(f"File uploaded successfully to mobile-price-range/models/model.joblib")
    except FileNotFoundError:
        print(f"The file {local_file_path} was not found.")
    except NoCredentialsError:
        print("Credentials not available.")

# Example usage
local_model_path = 'models/model.joblib'
s3_bucket_name = 'mobile-price-range'
s3_file_path = 'models/model.joblib'

upload_to_s3(local_model_path, s3_bucket_name, s3_file_path)
shutil.copy(local_model_path, 'model.joblib')
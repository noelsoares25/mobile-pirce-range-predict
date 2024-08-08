import boto3
import pandas as pd

s3 = boto3.client("s3")

s3=boto3.resource(
    service_name='s3',
    region_name='ap-south-1',
    aws_access_key_id='',
    aws_secret_access_key=''
)

obj = s3.Bucket('mobile-price-range').Object('raw/train.csv').get()
train = pd.read_csv(obj['Body'], index_col=0)
print(train)
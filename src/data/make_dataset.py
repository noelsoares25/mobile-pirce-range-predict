import boto3
import pandas as pd

def load_data():
    s3 = boto3.resource(
        service_name='s3',
        region_name='ap-south-1',
        aws_access_key_id='',
        aws_secret_access_key=''
    )

    # Load the data directly into a pandas DataFrame
    obj = s3.Bucket('mobile-price-range').Object('raw/train.csv').get()
    train = pd.read_csv(obj['Body'])
    return train


def main():
    # Call load_data and do something with the data if running as a script
    train = load_data()
    print(train.columns)
    # You can perform other operations here if needed

if __name__ == "__main__":
    main()
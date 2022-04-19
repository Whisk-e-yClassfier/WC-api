import boto3

"""
For Synchronous Events
"""


def s3_connection():
    '''
    s3 bucket에 연결
    :return: 연결된 s3 객체
    '''
    try:
        s3 = boto3.client(
            service_name='s3',
            region_name="ap-northeast-2",
            aws_access_key_id="your access key id",
            aws_secret_access_key="your secret access key"
        )
    except Exception as e:
        print(e)
        exit("ERROR_S3_CONNECTION_FAILED")
    else:
        print("s3 bucket connected!")
        return s3

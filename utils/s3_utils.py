# s3_utils.py
import os
import cv2
import numpy as np
import boto3
from botocore.exceptions import ClientError
from io import BytesIO
import logging
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_s3_client = None

def _get_s3_client():
    """Initializes and returns a singleton S3 client."""
    global _s3_client
    if _s3_client is None:
        aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        aws_region = os.environ.get('AWS_REGION')
        print(f"Using AWS Region: {aws_region}")
        if not all([aws_access_key_id, aws_secret_access_key, aws_region]):
            logger.error("AWS credentials or region not found in environment variables.")
            raise ValueError("AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION must be set as environment variables.")

        _s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        logger.info(f"S3 client initialized for region: {aws_region}")
    return _s3_client

def upload_image_array_to_s3(image_array: np.ndarray, bucket_name: str, s3_key: str) -> str:
    """
    Uploads a NumPy image array to an S3 bucket.
    :param image_array: NumPy array representing the image (e.g., from cv2.imread).
    :param bucket_name: Name of the S3 bucket.
    :param s3_key: Desired key (path/filename) for the object in S3.
    :return: The public URL of the uploaded image.
    :raises ClientError: If there's an issue with S3 upload.
    """
    s3 = _get_s3_client()
    try:
        # Encode the NumPy array to JPEG bytes in memory
        is_success, buffer = cv2.imencode(".jpg", image_array)
        if not is_success:
            raise ValueError("Could not encode image array to JPEG format.")

        byte_obj = BytesIO(buffer.tobytes())

        s3.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=byte_obj.getvalue(),
            ContentType='image/jpeg',
            # ACL='public-read' # Only include if your bucket policy allows/requires ACLs
        )
        s3_url = f"https://{bucket_name}.s3.{os.environ.get('AWS_REGION')}.amazonaws.com/{s3_key}"
        logger.info(f"Successfully uploaded image to S3: {s3_url}")
        return s3_url
    except ClientError as e:
        logger.error(f"S3 upload error for key '{s3_key}': {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during image upload: {e}")
        raise

def upload_file_to_s3(local_filepath: str, bucket_name: str, s3_key: str) -> str:
    """
    Uploads a local file to an S3 bucket.
    :param local_filepath: Path to the local file.
    :param bucket_name: Name of the S3 bucket.
    :param s3_key: Desired key (path/filename) for the object in S3.
    :return: The public URL of the uploaded file.
    :raises ClientError: If there's an issue with S3 upload.
    """
    s3 = _get_s3_client()
    try:
        s3.upload_file(local_filepath, bucket_name, s3_key)
        s3_url = f"https://{bucket_name}.s3.{os.environ.get('AWS_REGION')}.amazonaws.com/{s3_key}"
        logger.info(f"Successfully uploaded file to S3: {s3_url}")
        return s3_url
    except ClientError as e:
        logger.error(f"S3 file upload error for '{local_filepath}' to '{s3_key}': {e}")
        raise
    except FileNotFoundError:
        logger.error(f"Local file not found: {local_filepath}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during file upload: {e}")
        raise

def download_image_from_s3_url(image_url: str) -> np.ndarray:
    """
    Downloads an image from a given URL and returns it as a NumPy array.
    :param image_url: The URL of the image (e.g., S3 public URL).
    :return: NumPy array representing the image.
    :raises requests.exceptions.RequestException: If there's an issue downloading the image.
    :raises ValueError: If the image cannot be decoded.
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not decode image from URL: {image_url}")
        logger.info(f"Successfully downloaded and decoded image from URL: {image_url}")
        return img
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading image from URL '{image_url}': {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during image download/decode: {e}")
        raise

def download_file_from_s3(bucket_name: str, s3_key: str, local_filepath: str):
    """
    Downloads a file from S3 to a local path.
    :param bucket_name: Name of the S3 bucket.
    :param s3_key: Key of the object in S3.
    :param local_filepath: Local path to save the downloaded file.
    :raises ClientError: If there's an issue with S3 download.
    """
    s3 = _get_s3_client()
    try:
        s3.download_file(bucket_name, s3_key, local_filepath)
        logger.info(f"Successfully downloaded '{s3_key}' from S3 to '{local_filepath}'")
    except ClientError as e:
        logger.error(f"S3 download error for key '{s3_key}': {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during file download: {e}")
        raise

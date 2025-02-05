import os
import json
import time
import boto3
from botocore.exceptions import ClientError
from typing import Optional, Dict
from datetime import datetime
from botocore.config import Config
import signal
import sys
from predictor_class import PredictorService
from pathlib import Path
import tempfile
import base64
import cv2
import numpy as np
from ifnude import detect
from dotenv import load_dotenv

load_dotenv()


class DecentralizedDownloader:
    def __init__(self, 
                 aws_access_key: str,
                 aws_secret_key: str,
                 private_bucket_name: str,
                 public_bucket_name: str,
                 local_download_dir: str = 'temp_downloads',
                 config: Optional[Config] = None):
        """Initialize the decentralized downloader"""
        self.private_bucket_name = private_bucket_name
        self.public_bucket_name = public_bucket_name
        self.local_download_dir = local_download_dir
        
        # Ensure local directory exists
        os.makedirs(local_download_dir, exist_ok=True)
        
        # Initialize AWS client
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            config=config
        )
        
        # Initialize predictor service
        self.predictor = PredictorService
        
        self.state_file = Path('downloader_state.json')
        self.last_processed_id = self._load_last_processed_id()
        self.last_check_time = self._load_last_check_time()

    def _load_last_processed_id(self) -> int:
        """Load the last processed ID from state file"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    return state.get('last_processed_id', 0)
        except Exception as e:
            print(f"Error loading state file: {str(e)}")
        return 0

    def _load_last_check_time(self) -> float:
        """Load the last check timestamp from state file"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    return state.get('last_check_time', 0)
        except Exception as e:
            print(f"Error loading state file: {str(e)}")
        return 0

    def _save_state(self, last_id: int):
        """Save both last ID and check time to state file"""
        try:
            current_time = time.time()
            with open(self.state_file, 'w') as f:
                json.dump({
                    'last_processed_id': last_id,
                    'last_check_time': current_time
                }, f)
            self.last_check_time = current_time
        except Exception as e:
            print(f"Error saving state file: {str(e)}")

    def check_pending_downloads(self) -> list:
        """Check S3 for prediction JSONs that need processing"""
        try:
            print(f"\nChecking for pending predictions (after ID: {self.last_processed_id})...")
            pending_predictions = []
            all_contents = []
            
            # Use pagination to get all objects
            paginator = self.s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=self.private_bucket_name,
                Prefix='predictions/'
            )
            
            # Collect all contents, filtering for .json files first
            for page in pages:
                json_files = [obj for obj in page.get('Contents', []) 
                             if obj['Key'].endswith('.json')]
                all_contents.extend(json_files)
            
            # Sort objects by prediction ID numerically
            sorted_contents = sorted(
                all_contents,
                key=lambda x: int(x['Key'].split('/')[-1].replace('.json', '')),
                reverse=True  # Sort in descending order
            )

            print(f"Found {len(sorted_contents)} total JSON objects in S3")
            
            for obj in sorted_contents:
                try:
                    prediction_id = int(obj['Key'].split('/')[-1].replace('.json', ''))
                    print(f"Found prediction ID: {prediction_id}, last processed ID: {self.last_processed_id}")
                    
                    # Since we're sorted by ID in descending order, we can break if we hit a lower ID
                    if prediction_id <= self.last_processed_id:
                        print(f"Stopping at ID {prediction_id} as it's not newer than last processed ID {self.last_processed_id}")
                        break

                    json_obj = self.s3.get_object(Bucket=self.private_bucket_name, Key=obj['Key'])
                    prediction_data = json.loads(json_obj['Body'].read().decode('utf-8'))

                    print(f"Found prediction ID: {prediction_id}, status: {prediction_data.get('status')}")
                    
                    if prediction_data.get('status') == 'pending':
                        print(f"Found pending prediction ID: {prediction_id}")
                        pending_predictions.append(prediction_data)

                except ValueError:
                    print(f"Skipping invalid filename: {obj['Key']}")
                    continue
            
            print(f"\nFound {len(pending_predictions)} pending predictions")
            return pending_predictions
            
        except Exception as e:
            print(f"Error checking pending predictions: {str(e)}")
            return []

    def _check_image_safety(self, image_path: str, prediction_id: int) -> tuple[bool, str]:
        """
        Check if image is safe (no nudity) and valid
        Returns: (is_safe, error_message)
        """
        print(f"Checking image safety for prediction {prediction_id}", flush=True)
        try:
            # Convert bytes to numpy array for OpenCV
            # Detect nudity
            try:
                detections = detect(image_path, mode="fast")

                print(f"Detections: {detections}", flush=True)
                
                if detections:
                    detection_labels = [d['label'] for d in detections]
                    return False, f"NSFW content detected: {', '.join(detection_labels)}"
                    
                return True, ""
                
            except Exception as e:
                print(f"Error during nudity detection for prediction {prediction_id}: {str(e)}")
                return False, f"Error analyzing image content: {str(e)}"
                
        except Exception as e:
            print(f"Error processing image for prediction {prediction_id}: {str(e)}")
            return False, f"Error processing image: {str(e)}"

    def _delete_image_from_bucket(self, prediction_id: int):
        """Delete image from public bucket"""
        try:
            image_key = f"predictions/{prediction_id}.jpg"
            self.s3.delete_object(
                Bucket=self.public_bucket_name,
                Key=image_key
            )
            print(f"Successfully deleted unsafe image {image_key} from public bucket")
        except Exception as e:
            print(f"Error deleting image from public bucket: {str(e)}")

    def _save_temp_image(self, image_data: bytes) -> str:
        """
        Save image bytes to a temporary file and return the path
        Returns: path to temporary file
        """
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_file.write(image_data)
                return temp_file.name
        except Exception as e:
            print(f"Error saving temporary image: {str(e)}")
            raise

    def _cleanup_temp_file(self, temp_path: str):
        """Clean up temporary file"""
        try:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception as e:
            print(f"Error cleaning up temporary file: {str(e)}")

    def process_pending_downloads(self):
        """Process all pending predictions"""
        pending = self.check_pending_downloads()
        
        highest_id = self.last_processed_id

        print(f"Processing {len(pending)} pending predictions, starting from ID: {highest_id}")
        for prediction_data in pending:
            temp_path = None
            try:
                prediction_id = prediction_data.get('id')
                if not prediction_id:
                    continue

                print(f"\nProcessing prediction ID: {prediction_id}")

                # Get the image from public bucket
                try:
                    image_key = f"predictions/{prediction_id}.jpg"
                    print(f"Fetching image from public S3: {image_key}")
                    image_obj = self.s3.get_object(
                        Bucket=self.public_bucket_name,
                        Key=image_key
                    )
                    image_data = image_obj['Body'].read()
                    print(f"Successfully downloaded image, size: {len(image_data)} bytes")
                    
                    if len(image_data) == 0:
                        raise ValueError("Empty image file received from S3")
                    
                    # Save image to temporary file
                    temp_path = self._save_temp_image(image_data)
                    
                    # Check image safety
                    # is_safe, error_message = self._check_image_safety(temp_path, prediction_id)
                    if False:
                        print(f"Image safety check failed: {error_message}")
                        # Delete unsafe image
                        self._delete_image_from_bucket(prediction_id)
                        # Update prediction with error
                        prediction_data.update({
                            'status': 'error',
                            'error_message': "error_message",
                            'error_type': 'IMAGE_VALIDATION_ERROR',
                            'processed_at': datetime.utcnow().isoformat()
                        })
                        self.s3.put_object(
                            Bucket=self.private_bucket_name,
                            Key=f"predictions/{prediction_id}.json",
                            Body=json.dumps(prediction_data, indent=2),
                            ContentType='application/json'
                        )
                        continue

                    # Process the image using PredictorService
                    print(f"Starting prediction for image {prediction_id}")
                    print(f"prediction_data {prediction_data}")

                    # Check for image URL first
                    image_url = prediction_data.get('image_url')
                    if image_url:
                        print(f"Found image URL: {image_url}")
                        image_data = self.predictor.download_image(image_url)
                        if not image_data:
                            print(f"Failed to download image from URL: {image_url}")
                            continue

                    # Base64 encode the image data before passing to predictor
                    encoded_image = base64.b64encode(image_data)

                    # Extract keypoints from poses data
                    poses = prediction_data.get('poses', [])

                    camera_width = None
                    camera_height = None

                    if not poses:
                        print("No poses found, attempting to detect poses")
                        pose_results = self.predictor.process_image_posenet(image_data)
                        if pose_results:
                            poses = pose_results.get('poses', [])
                            print(f"Detected {len(poses)} poses")
                            camera_width = pose_results.get('camera_width', None)
                            camera_height = pose_results.get('camera_height', None)
                        else:
                            print("Failed to detect poses")

                    try:
                        measurement, age = self.predictor.predict_for_image(encoded_image, poses)
                        print(f"Prediction successful - measurement: {measurement}, age: {age}")
                    except Exception as e:
                        print(f"Error during prediction: {str(e)}")
                        measurement = 0.0
                        age = 0

                    # Update the prediction JSON with results

                    print(f"poses: {poses}", flush=True)


                    prediction_data.update({
                        'status': 'completed',
                        'measurement': round(measurement, 2),
                        'age': age,
                        'poses': poses,
                        'camera_width': camera_width,
                        'camera_height': camera_height,
                        'processed_at': datetime.utcnow().isoformat()
                    })

                    # Save updated JSON back to private bucket
                    json_content = json.dumps(prediction_data, indent=2)
                    print(f"json_content: {json_content}", flush=True)
                    self.s3.put_object(
                        Bucket=self.private_bucket_name,  # Use private bucket for JSONs
                        Key=f"predictions/{prediction_id}.json",
                        Body=json_content,
                        ContentType='application/json'
                    )

                    # Note: We don't delete the image from public bucket
                    print(f"Successfully processed prediction {prediction_id}")
                    highest_id = max(highest_id, prediction_id)

                except Exception as e:
                    print(f"Error processing prediction {prediction_id}: {str(e)}")

            finally:
                # Clean up temporary file
                if temp_path:
                    self._cleanup_temp_file(temp_path)

        # Update the last processed ID
        if highest_id > self.last_processed_id:
            print(f"Updating last processed ID to: {highest_id}", flush=True)
            self._save_state(highest_id)
            self.last_processed_id = highest_id
            print(f"Updated last processed ID to: {highest_id}", flush=True)

    def run_forever(self):
        """Run the processor in an infinite loop"""
        def signal_handler(signum, frame):
            print("\nReceived shutdown signal. Cleaning up...")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print(f"\nStarting prediction processor service...")
        
        while True:
            try:
                print("\n" + "="*50)
                print(f"Checking for pending predictions at {datetime.now().isoformat()}")
                self.process_pending_downloads()
                
            except Exception as e:
                raise e
                print(f"Error in main loop: {str(e)}")
                


def main():
    """Main entry point"""
    from deepface import DeepFace

    print("Loading AI models... This may take some time")
    DeepFace.build_model("Age")
    DeepFace.build_model("Gender")
    DeepFace.build_model("Race")

    try:
        # Initialize S3 client config
        my_config = Config(
            region_name="eu-west-1",
            signature_version="v4",
            retries={"max_attempts": 10, "mode": "standard"},
        )

        # Create the downloader
        downloader = DecentralizedDownloader(
            aws_access_key=os.getenv('AWS_ACCESS_KEY'),
            aws_secret_key=os.getenv('AWS_SECRET_KEY'),
            private_bucket_name=os.getenv('PRIVATE_BUCKET_NAME'),
            public_bucket_name=os.getenv('PUBLIC_BUCKET_NAME'),
            local_download_dir='temp_downloads',
            config=my_config
        )
        
        # Create temp directory if it doesn't exist
        os.makedirs('temp_downloads', exist_ok=True)
        
        try:
            downloader.run_forever()
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
        except Exception as e:
            print(f"Fatal error: {str(e)}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error initializing downloader: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
import requests
from requests.exceptions import RequestException
import logging
import os
import time
from PIL import Image
import argparse
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ServerTester:
    def __init__(self, base_url, image_path):
        self.base_url = base_url
        self.image_path = image_path
        self.session = requests.Session()

    def test_endpoints(self):
        results = {}

        results['image_check'] = self.check_image()

        endpoints = {
            '/test': 10,
            '/detect_fruit': 30
        }

        for endpoint, timeout in endpoints.items():
            results[endpoint] = self.test_endpoint(endpoint, timeout)

        return results

    def check_image(self):
        try:
            if not os.path.exists(self.image_path):
                return {"status": "failure", "error": f"Image not found: {self.image_path}"}

            with Image.open(self.image_path) as img:
                return {
                    "status": "success",
                    "details": {
                        "format": img.format,
                        "size": img.size,
                        "mode": img.mode,
                        "file_size": os.path.getsize(self.image_path)
                    }
                }
        except Exception as e:
            return {"status": "failure", "error": str(e)}

    def test_endpoint(self, endpoint, timeout):
        try:
            with open(self.image_path, 'rb') as img_file:
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}{endpoint}",
                    files={'image': img_file},
                    timeout=timeout
                )
                elapsed_time = time.time() - start_time

                return {
                    "status": "success" if response.status_code == 200 else "failure",
                    "status_code": response.status_code,
                    "elapsed_time": f"{elapsed_time:.2f} seconds",
                    "response": response.json() if response.status_code == 200 else None,
                    "error": response.text if response.status_code != 200 else None
                }
        except RequestException as e:
            return {"status": "failure", "error": str(e)}


def print_results(results, indent=0):
    for key, value in results.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_results(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


def main(base_url, image_path):
    try:
        tester = ServerTester(base_url, image_path)
        results = tester.test_endpoints()
        print("\n=== Test Results ===")
        print_results(results)
    except Exception as e:
        logger.error(f"An error occurred during testing: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test fruit detection server endpoints")
    parser.add_argument("--url", default="http://192.168.29.200:5005/", help="Base URL of the server")
    parser.add_argument("--image",
                        default="images/istockphoto-468098643-1024x1024 copy.jpg",
                        help="Path to the image file")

    args = parser.parse_args()

    main(args.url, args.image)
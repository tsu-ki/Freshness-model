import gdown
import os

def check_and_download_file(file_path, google_drive_url):
    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"{file_path} does not exist. Downloading from Google Drive...")

        # Download the file
        gdown.download(google_drive_url, file_path, quiet=False)
        print(f"Downloaded {file_path}.")
    else:
        print(f"{file_path} already exists.")


# Example usage
file_name = "fruit_model.h5"  # Replace with your desired file name
google_drive_url = "https://drive.google.com/uc?id=1GX46MFyLa5zI5X3L0SiWu4brCjcxMUes"  # Replace with your file's Google Drive URL
check_and_download_file(file_name, google_drive_url)
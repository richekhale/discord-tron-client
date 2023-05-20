from PIL import Image
import os
import csv
import shutil
import requests
import re
import sys

# Constants
FILE = "dataset.csv" # The CSV file to read data from
OUTPUT_DIR = "/home/kash/Downloads/datasets/midjourney" # Directory to save images

# Check if output directory exists, create if it does not
if not os.path.exists(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR)
    except Exception as e:
        print(f'Could not create output directory: {e}')
        sys.exit(1)

# Check if CSV file exists
if not os.path.exists(FILE):
    print(f'Could not find CSV file: {FILE}')
    sys.exit(1)

def content_to_filename(content):
    """
    Function to convert content to filename by stripping everything after '--', 
    replacing non-alphanumeric characters and spaces, converting to lowercase, 
    removing leading/trailing underscores, and limiting filename length to 128.
    """
    content = content.split('--', 1)[0] 
    cleaned_content = re.sub(r'[^a-zA-Z0-9 ]', '', content)
    cleaned_content = cleaned_content.replace(' ', '_').lower().strip('_')
    cleaned_content = cleaned_content[:128] if len(cleaned_content) > 128 else cleaned_content
    return cleaned_content + '.png'

def load_csv(file):
    """
    Function to load CSV data into a list of dictionaries
    """
    data = []
    with open(file, newline='') as csvfile:
        try:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
        except Exception as e:
            print(f'Could not advance reader: {e}')
    return data

def fetch_image(url, filename):
    """
    Function to fetch image from a URL and save it to disk if it is square
    """
    try:
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            current_file_path = os.path.join(OUTPUT_DIR, filename)
            with open(current_file_path, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
            image = Image.open(current_file_path)
            width, height = image.size
            if width != height:
                print(f'Image {filename} is not square ({width}x{height}), deleting...')
                os.remove(current_file_path)
                return
            print(f'Resizing image to {current_file_path}')
            image = image.resize((768, 768), resample=Image.LANCZOS)
            image.save(current_file_path, format='PNG')
            image.close()
        else:
            print(f'Could not fetch {filename} from {url} (status code {r.status_code})')
    except Exception as e:
        print(f'Could not fetch {filename} from {url}: {e}')

def fetch_data(data):
    """
    Function to fetch all images specified in data
    """
    to_fetch = {}
    for row in data:
        new_filename = content_to_filename(row['Content'])
        if "Variations" in row['Content'] or "Upscaled" not in row['Content']:
            continue
        if new_filename not in to_fetch:
            to_fetch[new_filename] = {'url': row['Attachments'], 'filename': new_filename}
    print(f'Fetching {len(to_fetch)} images...')
    for filename, info in to_fetch.items():
        print(f'Fetching {filename}...')
        fetch_image(info['url'], filename)

def main():
    """
    Main function to load CSV and fetch images
    """
    data = load_csv(FILE)
    fetch_data(data)

if __name__ == '__main__':
    main()
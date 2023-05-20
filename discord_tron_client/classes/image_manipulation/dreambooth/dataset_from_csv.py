from PIL import Image
# CSV has columns:
# AuthorID,Author,Date,Content,Attachments,Reactions
import os

file = "dataset.csv" # Replace this value with your correct CSV filename.
output_dir = "/home/kash/Downloads/datasets/midjourney"
# Create output_dir if it doesn't exist:
if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
    except Exception as e:
        print(f'Could not create output directory: {e}')
        exit(1)
if not os.path.exists(file):
    print(f'Could not find CSV file: {file}')
    exit(1)

import csv
import shutil
import sys
import time
import requests

import re

def content_to_filename(content):
    # Strip everything after the first '--', including the '--' itself
    content = content.split('--', 1)[0]
    # Strip all non-alphanumeric characters and spaces
    cleaned_content = re.sub(r'[^a-zA-Z0-9 ]', '', content)
    # Replace spaces with underscores
    cleaned_content = cleaned_content.replace(' ', '_')
    # Convert to lowercase
    cleaned_content = cleaned_content.lower()
    # Remove leading or trailing underscores
    cleaned_content = cleaned_content.strip('_')
    # Limit the filename length to 255 characters (typical UNIX filename limit)
    if len(cleaned_content) > 128:
        cleaned_content = cleaned_content[:128]
    return cleaned_content + '.png'

def main():
    # Load CSV into memory:
    data = load_csv(file)
    fetch_data(data)

def load_csv(file):
    data = []
    with open(file, newline='') as csvfile:
        try:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
        except Exception as e:
            print(f'Could not advance reader: {e}')
            return data

    return data

def fetch_image(url, filename):
    try:
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(os.path.join(output_dir, filename), 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
        else:
            print(f'Could not fetch {filename} from {url} (status code {r.status_code})')

        image = Image.open(os.path.join(output_dir, filename))
        width, height = image.size
        image.close()
        if width != height:
            print(f'Image {filename} ({type}) is not square ({width}x{height})')
            # Delete the image:
            os.remove(os.path.join(output_dir, filename))

    except Exception as e:
        print(f'Could not fetch {filename} from {url}: {e}')

def fetch_data(data: list):
    to_fetch = {}
    count_total = len(data)
    count_to_fetch = 0
    for row in data:
        new_filename = content_to_filename(row['Content'])
        type = None
        if "Variations" in row['Content']:
            type = 'variations'
            continue
        elif "Upscaled" in row['Content']:
            type = 'upscale'
        else:
            type = 'original'
            continue
        if new_filename in to_fetch:
            # to_fetch[new_filename]['extra_urls'][type] = row['Attachments']
            pass
        else:
            to_fetch[new_filename] = {
                'url': row['Attachments'],
                'filename': new_filename,
                'extra_urls': {}
            }
        count_to_fetch += 1
    print(f'Fetching {count_to_fetch} of {count_total} images...')
    count_fetched = 0
    for filename, info in to_fetch.items():
        print(f'Fetching {filename}...')
        fetch_image(info['url'], filename)
        count_fetched += 1
        if info['extra_urls'] is not None:
            for type, url in info['extra_urls'].items():
                print(f'Fetching {filename} ({type})...')
                fetch_image(url, f'{count_fetched}_{filename}')
                count_fetched += 1
        print(f'Fetched {count_fetched} of {count_to_fetch} images...')

if __name__ == '__main__':
    main()
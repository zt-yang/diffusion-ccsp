import gdown
import os
from os.path import isdir, isfile, join
import zipfile

urls = {
    'checkpoints': 'https://drive.google.com/drive/folders/1UhFNPGR0VQ9Ai-VtlHsMYWd3PdHKkGVR?usp=sharing',
    'wandb': 'https://drive.google.com/drive/folders/1cOibF5GiiyB5tDixcUCQOOvVkyL3cn4j?usp=drive_link',
    'data': 'https://drive.google.com/drive/folders/172y1OhG-7t_SVgVTEXQjh0eJz3-n3h2d?usp=drive_link'
}

for name, url in urls.items():
    print('Downloading', name)
    gdown.download_folder(url, quiet=True, use_cookies=False)

## unzip data folders
data_names = [join('data', f) for f in os.listdir('data') if f.endswith('.zip')]
for data_name in data_names:
    with zipfile.ZipFile(data_name, 'r') as zip_ref:
        zip_ref.extractall('data')

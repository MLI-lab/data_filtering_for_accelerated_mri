import os
from glob import glob
from pathlib import Path
import yaml
import json

if __name__ == '__main__':
    config_file = './datasets/path_to_datasets.yml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    files = glob('./datasets/**/*.json', recursive=True)
    for file in files:
        try:
            with open(file) as f:
                json_data = json.load(f)
        
            for filename, data in json_data['files'].items():
                old_path_to_file = data['path']
                dataset_name = data['dataset_name']
                split = data['split']
                path_to_dataset = config[split][dataset_name]
                new_path_to_file = os.path.join(path_to_dataset, filename)

                json_data['files'][filename]['path'] = new_path_to_file

            json_data = json.dumps(json_data, indent=4)
            with open(file, "w") as outfile:
                outfile.write(json_data)
        except Exception as e:
            print(e)


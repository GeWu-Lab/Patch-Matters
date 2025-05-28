import json
import os
import argparse

# Set up command line argument parsing
parser = argparse.ArgumentParser(description='Merge all JSON files in the specified folder')
parser.add_argument('--folder_path', type=str, help='Path to the folder containing JSON files')
parser.add_argument('--output_file', type=str, help='Path for the output merged JSON file')

args = parser.parse_args()

# Get all JSON files in the specified folder
json_files = [f for f in os.listdir(args.folder_path) if f.endswith('.json')]
print(json_files)
# List to store merged data
merged_data = []

# Iterate through each file and merge its contents into the same list
for file in json_files:
    file_path = os.path.join(args.folder_path, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):  # Ensure each file's data is a list
            merged_data.extend(data)  # Merge the list data into merged_data

# Save the merged data to a new JSON file
with open(args.output_file, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

print(f"Merging complete, results saved to {args.output_file}")

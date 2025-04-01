import csv
import glob
import re

def process_csv(file_path):
    # List to store extracted second IDs
    extracted_ids = []
    
    # Compile regex pattern to extract the second id from matched_pairs, e.g., "(1, 290)"
    pattern = re.compile(r'\(\s*\d+\s*,\s*(\d+)\s*\)')
    
    # Open the CSV file using csv.DictReader
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                # Convert matched_scores to float
                score = float(row['matched_scores'])
            except ValueError:
                continue  # Skip row if conversion fails
                
            # Process rows where matched_scores is less than 0.5
            if score < 0.5:
                pair = row['matched_pairs']
                # Use regex to find the second id in the matched_pairs string
                match = pattern.search(pair)
                if match:
                    extracted_ids.append(match.group(1))
                    
    # Remove duplicates by converting the list to a set, then back to a list
    unique_ids = list(set(extracted_ids))
    return unique_ids

def process_folder(folder_path):
    # Find all CSV files in the specified folder
    csv_files = glob.glob(folder_path + "/*.csv")
    results = {}
    for file in csv_files:
        unique_ids = process_csv(file)
        results[file] = {
            'id_list': unique_ids,
            'count': len(unique_ids)
        }
    return results

# Example usage:
# Set folder_path to the path containing your CSV files
folder_path = "/mmfs1/data/liupen/project/dataset/nuclei/long_2009/proofreaded"
results = process_folder(folder_path)

# Print the unique ID list and count for each file
for file, data in results.items():
    print(f"File: {file}")
    print(f"Unique IDs: {data['id_list']}")
    print(f"Count: {data['count']}")
    print("-" * 40)

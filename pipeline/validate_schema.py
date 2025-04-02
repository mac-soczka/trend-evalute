import os
import json

def load_features_schema(filepath='features_schema.json'):
    with open(filepath, 'r') as f:
        schema = json.load(f)
    return [feature["name"] for feature in schema]

def validate_data_files(data_folder='data/raw', schema_filepath='features_schema.json'):
    invalid_files = []
    required_features = load_features_schema(schema_filepath)
    
    for file_name in os.listdir(data_folder):
        if file_name.endswith(".json"):
            file_path = os.path.join(data_folder, file_name)
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                if isinstance(data, dict) and "features" in data:
                    data = data["features"]
                elif isinstance(data, list):
                    pass
                else:
                    print(f"Unexpected structure in {file_name}, skipping validation.")
                    continue
                
                for idx, datapoint in enumerate(data):
                    missing_features = [feature for feature in required_features if feature not in datapoint]
                    if missing_features:
                        print(f"\nInvalid datapoint in file: {file_name}, index: {idx}")
                        print(f"Missing features: {', '.join(missing_features)}")
                        print(f"Datapoint content: {json.dumps(datapoint, indent=2)}")
                        invalid_files.append({
                            "file": file_name,
                            "index": idx,
                            "missing_features": missing_features
                        })

    return invalid_files

invalid_files = validate_data_files()

if not invalid_files:
    print("All files contain the required features.")

import os
import json

# Load the features schema from the features_schema.json file
def load_features_schema(filepath='features_schema.json'):
    with open(filepath, 'r') as f:
        schema = json.load(f)
    # Extract the feature names from the schema
    print("Loaded features schema:")
    #print(schema)  # Printing schema for debugging
    return [feature["name"] for feature in schema]

# Function to validate raw data files
def validate_data_files(data_folder='data/raw', schema_filepath='features_schema.json'):
    invalid_files = []
    required_features = load_features_schema(schema_filepath)
    
    for file_name in os.listdir(data_folder):
        if file_name.endswith(".json"):
            file_path = os.path.join(data_folder, file_name)
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Debugging: Print the structure of the data
                print(f"Structure of {file_name}:")
                #print(data)  # This will show how the data is structured
                
                # Check if the data has a 'features' key or just a list of feature dictionaries
                if isinstance(data, dict) and "features" in data:
                    data = data["features"]
                elif isinstance(data, list):  # If it's already a list of features, no need to extract 'features'
                    pass
                else:
                    print(f"Unexpected structure in {file_name}, skipping validation.")
                    continue
                
                # Check if the data has all required features
                feature_names = [feature["name"] for feature in data if "name" in feature]  # Ensuring 'name' exists
                missing_features = [feature for feature in required_features if feature not in feature_names]
                
                if missing_features:
                    invalid_files.append({
                        "file": file_name,
                        "missing_features": missing_features
                    })

    return invalid_files

# Run validation
invalid_files = validate_data_files()

if invalid_files:
    for file in invalid_files:
        print(f"File {file['file']} is missing features: {', '.join(file['missing_features'])}")
else:
    print("All files contain the required features.")

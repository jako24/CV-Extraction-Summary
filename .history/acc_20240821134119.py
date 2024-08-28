import json
from difflib import SequenceMatcher
import pandas as pd

# Define which fields require exact matches
exact_match_fields = {"name", "email", "phone", "graduation_date", "start_date", "end_date", "total_experience"}

# Define which fields use sequence matching for similarity
sequence_match_fields = {"degree", "branch", "institution", "technical_skills", "soft_skills"}

# Function to calculate exact match accuracy
def exact_match_accuracy(extracted_value, correct_value):
    return round(100.0 if extracted_value == correct_value else 0.0, 2)

# Function to calculate sequence match accuracy
def sequence_match_accuracy(extracted_value, correct_value):
    if isinstance(extracted_value, dict):
        extracted_value = json.dumps(extracted_value)
    if isinstance(correct_value, dict):
        correct_value = json.dumps(correct_value)
    if extracted_value is None:
        extracted_value = ""
    if correct_value is None:
        correct_value = ""
    return round(SequenceMatcher(None, extracted_value, correct_value).ratio() * 100, 2)

# Function to calculate attribute accuracy based on the type of field
def calculate_attribute_accuracy(extracted_values, correct_values, attribute):
    accuracies = []
    for correct_value in correct_values:
        if attribute in exact_match_fields:
            match_accuracies = [exact_match_accuracy(extracted_value, correct_value) for extracted_value in extracted_values]
        else:
            match_accuracies = [sequence_match_accuracy(extracted_value, correct_value) for extracted_value in extracted_values]
        
        if match_accuracies:
            best_match_accuracy = max(match_accuracies)
            accuracies.append(best_match_accuracy)
    
    if accuracies:
        return round(sum(accuracies) / len(accuracies), 2)
    return 0.0

# Function to handle nested JSON extraction for education and work experience
def nested_accuracy(extracted_entries, correct_entries, attribute):
    if not extracted_entries or not correct_entries:
        return 0.0
    
    accuracies = []
    
    for correct_entry in correct_entries:
        entry_accuracies = []
        for key in correct_entry.keys():
            if key in extracted_entries[0]:  # Ensure key exists in extracted entries
                entry_accuracy = calculate_attribute_accuracy(
                    [entry[key] for entry in extracted_entries if key in entry], 
                    [correct_entry[key]], 
                    key
                )
                entry_accuracies.append(entry_accuracy)
        
        if entry_accuracies:
            accuracies.append(sum(entry_accuracies) / len(entry_accuracies))
    
    return round(sum(accuracies) / len(accuracies), 2) if accuracies else 0.0

# Calculate overall accuracy for the given data
def calculate_overall_accuracy(extracted_data, correct_data):
    accuracies = []
    
    for key in correct_data.keys():
        if key in {"education", "work_experience"}:
            accuracy = nested_accuracy(extracted_data.get(key, []), correct_data.get(key, []), key)
        elif key in {"skills"}:
            tech_skills_accuracy = sequence_match_accuracy(
                " ".join(extracted_data.get(key, {}).get("technical_skills", [])),
                " ".join(correct_data.get(key, {}).get("technical_skills", []))
            )
            soft_skills_accuracy = sequence_match_accuracy(
                " ".join(extracted_data.get(key, {}).get("soft_skills", [])),
                " ".join(correct_data.get(key, {}).get("soft_skills", []))
            )
            accuracy = (tech_skills_accuracy + soft_skills_accuracy) / 2
        else:
            accuracy = calculate_attribute_accuracy(
                [extracted_data.get(key, "")], [correct_data[key]], key
            )
        accuracies.append(accuracy)
    
    if accuracies:
        return round(sum(accuracies) / len(accuracies), 2)
    return 0.0

# Function to calculate and save accuracies to the file without modifying any other columns
def calculate_extraction_accuracy(file_path):
    # Determine if the file is CSV or Excel based on the file extension
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, engine='openpyxl')
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path, on_bad_lines='skip', encoding='utf-8')
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    
    # Ensure columns exist
    required_columns = {'PDFLoaderExtraction', 'OCRExtraction', 'CorrectExtraction', 
                        'CombinedOCRPDFExtraction', 'EasyOCRExtraction'}
    if not required_columns.issubset(df.columns):
        raise ValueError("The file does not contain the required columns.")
    
    # Calculate accuracies for each row using the different extraction methods
    def safe_json_loads(data):
        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            return {}

    # Calculate LoaderAccuracy
    df['LoaderAccuracy'] = df.apply(lambda row: calculate_overall_accuracy(
        safe_json_loads(row['PDFLoaderExtraction'].lower()), 
        safe_json_loads(row['CorrectExtraction'].lower())
    ), axis=1)

    # Calculate OCRAccuracy
    df['OCRAccuracy'] = df.apply(lambda row: calculate_overall_accuracy(
        safe_json_loads(row['OCRExtraction'].lower()), 
        safe_json_loads(row['CorrectExtraction'].lower())
    ), axis=1)
    
    # Calculate CombinedOCRAccuracy
    df['CombinedOCRAccuracy'] = df.apply(lambda row: calculate_overall_accuracy(
        safe_json_loads(row['CombinedOCRPDFExtraction'].lower()), 
        safe_json_loads(row['CorrectExtraction'].lower())
    ), axis=1)
    
    # Calculate EasyOCRAccuracy
    df['EasyOCRAccuracy'] = df.apply(lambda row: calculate_overall_accuracy(
        safe_json_loads(row['EasyOCRExtraction'].lower()), 
        safe_json_loads(row['CorrectExtraction'].lower())
    ), axis=1)
    
    # Save the updated data back to the file
    if file_path.endswith('.xlsx'):
        df.to_excel(file_path, index=False, engine='openpyxl')
    elif file_path.endswith('.csv'):
        df.to_csv(file_path, index=False, encoding='utf-8')

    return df

# Usage example
csv_path = '/Users/janekkorczynski/CVExtraction/final_accuracy_check.csv'
updated_df = calculate_extraction_accuracy(csv_path)

# Example usage:
# csv_path = '/Users/janekkorczynski/CVExtraction/test_cv_extraction - extracted_information_test_cv copy.csv'
# calculate_extraction_accuracy(csv_path)
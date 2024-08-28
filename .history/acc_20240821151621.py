import json
from difflib import SequenceMatcher
import pandas as pd

# Define which fields require exact matches
exact_match_fields = {"name", "email", "phone", "graduation_date", "start_date", "end_date", "total_experience"}

# Define which fields use sequence matching for similarity
sequence_match_fields = {"degree", "branch", "institution", "technical_skills", "soft_skills"}

# Function to normalize and clean the string data
def normalize_string(value):
    if isinstance(value, str):
        return value.strip().lower().replace("-", "").replace(",", "").replace(".", "")
    return str(value).strip().lower()

def exact_match_accuracy(extracted_value, correct_value):
    if not extracted_value or not correct_value:
        print(f"Empty or None value found. Extracted: {extracted_value}, Correct: {correct_value}")
        return 0.0
    # Normalize before comparison
    extracted_value = normalize_string(extracted_value)
    correct_value = normalize_string(correct_value)
    
    # Log for debugging
    print(f"Comparing Exact Match - Extracted: '{extracted_value}', Correct: '{correct_value}'")
    
    return round(100.0 if extracted_value == correct_value else 0.0, 2)

def sequence_match_accuracy(extracted_value, correct_value):
    if not extracted_value or not correct_value:
        print(f"Empty or None value found. Extracted: {extracted_value}, Correct: {correct_value}")
        return 0.0

    # Normalize strings
    extracted_value = normalize_string(extracted_value)
    correct_value = normalize_string(correct_value)
    
    # Log for debugging
    print(f"Comparing Sequence Match - Extracted: '{extracted_value}', Correct: '{correct_value}'")

    # Perform the sequence matching
    return round(SequenceMatcher(None, extracted_value, correct_value).ratio() * 100, 2)

# Function to handle nested JSON extraction for education and work experience
def nested_accuracy(extracted_entries, correct_entries, attribute):
    if not extracted_entries or not correct_entries:
        print(f"No entries found for {attribute}. Extracted: {extracted_entries}, Correct: {correct_entries}")
        return 0.0
    
    accuracies = []
    
    for correct_entry in correct_entries:
        entry_accuracies = []
        for key, correct_value in correct_entry.items():
            extracted_values = [entry.get(key, "") for entry in extracted_entries if key in entry]
            
            # Log extracted and correct entries
            print(f"Comparing nested key '{key}' - Extracted: {extracted_values}, Correct: {correct_value}")
            
            if key in exact_match_fields or key in sequence_match_fields:
                entry_accuracy = calculate_attribute_accuracy(extracted_values, [correct_value], key)
                entry_accuracies.append(entry_accuracy)
        
        if entry_accuracies:
            accuracies.append(sum(entry_accuracies) / len(entry_accuracies))
    
    return round(sum(accuracies) / len(accuracies), 2) if accuracies else 0.0


# Function to calculate attribute accuracy based on the type of field
def calculate_attribute_accuracy(extracted_values, correct_values, attribute):
    accuracies = []

    if not isinstance(extracted_values, list):
        extracted_values = [extracted_values]
    if not isinstance(correct_values, list):
        correct_values = [correct_values]

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

# Function to calculate overall accuracy for the given data
def calculate_overall_accuracy(extracted_data, correct_data):
    accuracies = []
    
    for key in correct_data.keys():
        print(f"Processing key: {key}")
        
        if key in {"education", "work_experience"}:
            accuracy = nested_accuracy(extracted_data.get(key, []), correct_data.get(key, []), key)
        elif key == "skills":
            # Compare skills separately
            tech_skills_accuracy = sequence_match_accuracy(
                " ".join(normalize_string(skill) for skill in extracted_data.get(key, {}).get("technical_skills", [])),
                " ".join(normalize_string(skill) for skill in correct_data.get(key, {}).get("technical_skills", []))
            )
            soft_skills_accuracy = sequence_match_accuracy(
                " ".join(normalize_string(skill) for skill in extracted_data.get(key, {}).get("soft_skills", [])),
                " ".join(normalize_string(skill) for skill in correct_data.get(key, {}).get("soft_skills", []))
            )
            accuracy = (tech_skills_accuracy + soft_skills_accuracy) / 2
        elif key == "total_experience":
            years_accuracy = exact_match_accuracy(extracted_data.get(key, {}).get("years", ""), correct_data.get(key, {}).get("years", ""))
            months_accuracy = exact_match_accuracy(extracted_data.get(key, {}).get("months", ""), correct_data.get(key, {}).get("months", ""))
            accuracy = (years_accuracy + months_accuracy) / 2
        else:
            accuracy = calculate_attribute_accuracy([extracted_data.get(key, "")], [correct_data[key]], key)
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
        except json.JSONDecodeError:
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
csv_path = '/Users/janekkorczynski/CVExtraction/TestAccuracy.csv'
updated_df = calculate_extraction_accuracy(csv_path)

# Example usage:
# csv_path = '/Users/janekkorczynski/CVExtraction/test_cv_extraction - extracted_information_test_cv copy.csv'
# calculate_extraction_accuracy(csv_path)
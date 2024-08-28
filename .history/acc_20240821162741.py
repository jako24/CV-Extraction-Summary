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
        normalized_value = value.strip().lower().replace("-", "").replace(",", "").replace(".", "")
        print(f"Normalized Value: '{normalized_value}' from '{value}'")
        return normalized_value
    return str(value).strip().lower()

# Function to handle exact match accuracy
def exact_match_accuracy(extracted_value, correct_value):
    print(f"Comparing exact match - Extracted: {extracted_value}, Correct: {correct_value}")
    if not extracted_value or not correct_value:
        print(f"Exact match failure: Extracted: '{extracted_value}', Correct: '{correct_value}'")
        return 0.0
    # Normalize before comparison
    extracted_value = normalize_string(extracted_value)
    correct_value = normalize_string(correct_value)
    match_result = round(100.0 if extracted_value == correct_value else 0.0, 2)
    print(f"Exact Match: Extracted: '{extracted_value}', Correct: '{correct_value}', Match Result: {match_result}")
    return match_result


# Function to calculate sequence match accuracy
def sequence_match_accuracy(extracted_value, correct_value):
    if not extracted_value or not correct_value:
        print(f"Sequence match failure: Extracted: '{extracted_value}', Correct: '{correct_value}'")
        return 0.0

    # Normalize strings
    extracted_value = normalize_string(extracted_value)
    correct_value = normalize_string(correct_value)
    
    # Perform the sequence matching
    match_ratio = SequenceMatcher(None, extracted_value, correct_value).ratio()
    match_result = round(match_ratio * 100, 2)
    print(f"Sequence Match: Extracted: '{extracted_value}', Correct: '{correct_value}', Match Ratio: {match_ratio}, Match Result: {match_result}")
    return match_result

# Function to handle comparison of JSON objects like contact_info
def compare_contact_info(extracted_value, correct_value):
    # Ensure extracted_value and correct_value are dictionaries
    if isinstance(extracted_value, str):
        try:
            extracted_value = json.loads(extracted_value)
        except json.JSONDecodeError:
            print(f"Failed to parse extracted_value as JSON: {extracted_value}")
            extracted_value = {}
    
    if isinstance(correct_value, str):
        try:
            correct_value = json.loads(correct_value)
        except json.JSONDecodeError:
            print(f"Failed to parse correct_value as JSON: {correct_value}")
            correct_value = {}

    # Extract the email and phone fields, handling missing keys
    extracted_email = extracted_value.get("email", "")
    correct_email = correct_value.get("email", "")
    extracted_phone = extracted_value.get("phone", "")
    correct_phone = correct_value.get("phone", "")
    
    # Calculate accuracy for email and phone fields
    email_accuracy = exact_match_accuracy(extracted_email, correct_email)
    phone_accuracy = exact_match_accuracy(extracted_phone, correct_phone)
    
    # Log the accuracy results for contact_info
    print(f"Contact Info Accuracy - Email: {email_accuracy}, Phone: {phone_accuracy}")
    
    # Return the average accuracy of email and phone
    return (email_accuracy + phone_accuracy) / 2


# Revised skills comparison by tokenizing the strings
def tokenize_and_compare_skills(extracted_skills, correct_skills):
    extracted_tokens = set(extracted_skills.lower().split())
    correct_tokens = set(correct_skills.lower().split())
    
    common_tokens = extracted_tokens.intersection(correct_tokens)
    total_tokens = len(correct_tokens)
    
    accuracy = round(len(common_tokens) / total_tokens * 100, 2) if total_tokens > 0 else 0.0
    print(f"Tokenized Skills Match: Extracted Tokens: {extracted_tokens}, Correct Tokens: {correct_tokens}, Accuracy: {accuracy}")
    return accuracy

# Calculate skills accuracy separately using tokenization
def calculate_skills_accuracy(extracted_data, correct_data):
    extracted_tech_skills = " ".join(extracted_data.get("technical_skills", []))
    correct_tech_skills = " ".join(correct_data.get("technical_skills", []))
    tech_skills_accuracy = tokenize_and_compare_skills(extracted_tech_skills, correct_tech_skills)
    
    extracted_soft_skills = " ".join(extracted_data.get("soft_skills", []))
    correct_soft_skills = " ".join(correct_data.get("soft_skills", []))
    soft_skills_accuracy = tokenize_and_compare_skills(extracted_soft_skills, correct_soft_skills)
    
    total_accuracy = (tech_skills_accuracy + soft_skills_accuracy) / 2
    print(f"Skills Accuracy - Tech: {tech_skills_accuracy}, Soft: {soft_skills_accuracy}, Total: {total_accuracy}")
    return total_accuracy

# Fix the total_experience comparison to handle '0' correctly
def calculate_total_experience_accuracy(extracted_data, correct_data):
    years_accuracy = exact_match_accuracy(extracted_data.get("years", 0), correct_data.get("years", 0))
    months_accuracy = exact_match_accuracy(extracted_data.get("months", 0), correct_data.get("months", 0))
    
    total_experience_accuracy = (years_accuracy + months_accuracy) / 2
    print(f"Total Experience Accuracy - Years: {years_accuracy}, Months: {months_accuracy}, Total: {total_experience_accuracy}")
    return total_experience_accuracy

# Function to calculate attribute accuracy based on the type of field
def calculate_attribute_accuracy(extracted_values, correct_values, attribute):
    accuracies = []

    if not isinstance(extracted_values, list):
        extracted_values = [extracted_values]
    if not isinstance(correct_values, list):
        correct_values = [correct_values]

    for correct_value in correct_values:
        if attribute == "contact_info":
            # Handle contact_info as a JSON comparison
            match_accuracies = [compare_contact_info(extracted_value, correct_value) for extracted_value in extracted_values]
        elif attribute in exact_match_fields:
            match_accuracies = [exact_match_accuracy(extracted_value, correct_value) for extracted_value in extracted_values]
        else:
            match_accuracies = [sequence_match_accuracy(extracted_value, correct_value) for extracted_value in extracted_values]
        
        if match_accuracies:
            best_match_accuracy = max(match_accuracies)
            accuracies.append(best_match_accuracy)
    
    if accuracies:
        overall_accuracy = round(sum(accuracies) / len(accuracies), 2)
        print(f"Attribute Accuracy - {attribute}: {overall_accuracy}")
        return overall_accuracy
    return 0.0

# Function to handle nested JSON extraction for education and work experience
def nested_accuracy(extracted_entries, correct_entries, attribute):
    if not extracted_entries or not correct_entries:
        print(f"No entries found for {attribute}. Extracted: {extracted_entries}, Correct: {correct_entries}")
        return 0.0
    
    accuracies = []

    # Align each correct entry with the best matching extracted entry
    for correct_entry in correct_entries:
        entry_accuracies = []
        for key, correct_value in correct_entry.items():
            extracted_values = [entry.get(key, "") for entry in extracted_entries if key in entry]
            
            if key in exact_match_fields or key in sequence_match_fields:
                entry_accuracy = calculate_attribute_accuracy(extracted_values, [correct_value], key)
                entry_accuracies.append(entry_accuracy)
        
        if entry_accuracies:
            accuracy = sum(entry_accuracies) / len(entry_accuracies)
            print(f"Nested Accuracy - {attribute}: {accuracy}")
            accuracies.append(accuracy)
    
    return round(sum(accuracies) / len(accuracies), 2) if accuracies else 0.0

# Function to calculate overall accuracy for the given data
def calculate_overall_accuracy(extracted_data, correct_data):
    accuracies = []
    
    for key in correct_data.keys():
        print(f"Processing key: {key}")
        
        if key in {"education", "work_experience"}:
            accuracy = nested_accuracy(extracted_data.get(key, []), correct_data.get(key, []), key)
        elif key == "skills":
            accuracy = calculate_skills_accuracy(extracted_data.get(key, {}), correct_data.get(key, {}))
        elif key == "total_experience":
            accuracy = calculate_total_experience_accuracy(extracted_data.get(key, {}), correct_data.get(key, {}))
        else:
            accuracy = calculate_attribute_accuracy(extracted_data.get(key, ""), correct_data[key], key)
        
        accuracies.append(accuracy)
        print(f"Accuracy for {key}: {accuracy}")
    
    if accuracies:
        overall_accuracy = round(sum(accuracies) / len(accuracies), 2)
        print(f"Overall Accuracy: {overall_accuracy}")
        return overall_accuracy
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
    
    # Function to safely load JSON data
    def safe_json_loads(data):
        if isinstance(data, str):
            try:
                loaded_data = json.loads(data)
                return loaded_data
            except json.JSONDecodeError:
                print(f"Failed to parse JSON: {data}")
                return {}
        return {}

    # Calculate accuracies for each method
    df['LoaderAccuracy'] = df.apply(lambda row: calculate_overall_accuracy(
        safe_json_loads(row['PDFLoaderExtraction']), 
        safe_json_loads(row['CorrectExtraction'])
    ), axis=1)
    
    df['OCRAccuracy'] = df.apply(lambda row: calculate_overall_accuracy(
        safe_json_loads(row['OCRExtraction']), 
        safe_json_loads(row['CorrectExtraction'])
    ), axis=1)
    
    df['CombinedOCRAccuracy'] = df.apply(lambda row: calculate_overall_accuracy(
        safe_json_loads(row['CombinedOCRPDFExtraction']), 
        safe_json_loads(row['CorrectExtraction'])
    ), axis=1)

    df['EasyOCRAccuracy'] = df.apply(lambda row: calculate_overall_accuracy(
        safe_json_loads(row['EasyOCRExtraction']), 
        safe_json_loads(row['CorrectExtraction'])
    ), axis=1)
    
    # Calculate mean accuracies
    mean_loader_accuracy = df['LoaderAccuracy'].mean()
    mean_ocr_accuracy = df['OCRAccuracy'].mean()
    mean_combined_ocr_accuracy = df['CombinedOCRAccuracy'].mean()
    mean_easyocr_accuracy = df['EasyOCRAccuracy'].mean()

    # Store the mean accuracies in a new row
    summary_row = {
        'LoaderAccuracy': round(mean_loader_accuracy, 2),
        'OCRAccuracy': round(mean_ocr_accuracy, 2),
        'CombinedOCRAccuracy': round(mean_combined_ocr_accuracy, 2),
        'EasyOCRAccuracy': round(mean_easyocr_accuracy, 2)
    }
    
    # Determine the best accuracy method
    best_method = max(summary_row, key=summary_row.get)
    
    # Add the best method to the summary row
    summary_row['BestMethod'] = best_method
    
    # Convert the summary row to a DataFrame
    summary_df = pd.DataFrame([summary_row])

    # Concatenate the summary row to the existing dataframe
    df = pd.concat([df, summary_df], ignore_index=True)
    
    # Save the updated DataFrame back to a CSV or Excel file
    if file_path.endswith('.xlsx'):
        df.to_excel(file_path, index=False, engine='openpyxl')
    elif file_path.endswith('.csv'):
        df.to_csv(file_path, index=False, encoding='utf-8')

    return df
# Usage example
csv_path = '/Users/janekkorczynski/CVExtraction/final_accuracy.csv'
updated_df = calculate_extraction_accuracy(csv_path)

# Example usage:
# csv_path = '/Users/janekkorczynski/CVExtraction/test_cv_extraction - extracted_information_test_cv copy.csv'
# calculate_extraction_accuracy(csv_path)
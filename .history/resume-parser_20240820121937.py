import streamlit as st
import tempfile
import os
import json
from pyresparser import ResumeParser
import pandas as pd
import logging
from datetime import datetime
from dateutil import parser
from dateutil.relativedelta import relativedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_resume_info(file_path):
    try:
        data = ResumeParser(file_path).get_extracted_data()
        return data
    except Exception as e:
        logger.error(f"Error extracting data from {file_path}: {str(e)}")
        return None

def process_education(education_list):
    processed_education = []
    for edu in education_list:
        education_item = {
            "degree": "",
            "branch": "",
            "institution": "",
            "graduation_date": ""
        }
        if isinstance(edu, str):
            education_item["institution"] = edu
        elif isinstance(edu, dict):
            education_item["degree"] = edu.get("degree", "")
            education_item["institution"] = edu.get("school", "")
            education_item["graduation_date"] = edu.get("year", "")
        processed_education.append(education_item)
    return processed_education

def process_experience(experience_list):
    processed_experience = []
    for exp in experience_list:
        experience_item = {
            "company_name": "",
            "job_title": "",
            "start_date": "",
            "end_date": ""
        }
        if isinstance(exp, str):
            experience_item["company_name"] = exp
        elif isinstance(exp, dict):
            experience_item["company_name"] = exp.get("company", "")
            experience_item["job_title"] = exp.get("position", "")
            experience_item["start_date"] = exp.get("start_date", "")
            experience_item["end_date"] = exp.get("end_date", "")
        processed_experience.append(experience_item)
    return processed_experience

def calculate_total_experience(work_experience):
    total_experience = relativedelta()
    for job in work_experience:
        try:
            start_date = parser.parse(job['start_date'])
            end_date_str = job['end_date'].lower()
            if end_date_str in {'present', 'now', 'current', 'ongoing', 'till now'}:
                end_date = datetime.now()
            else:
                end_date = parser.parse(job['end_date'])
            total_experience += relativedelta(end_date, start_date)
        except Exception as e:
            logger.warning(f"Error parsing dates for job: {job}. Error: {str(e)}")
    return total_experience.years, total_experience.months

def format_extracted_info(raw_data):
    formatted_data = {
        "name": raw_data.get("name", ""),
        "contact_info": {
            "email": raw_data.get("email", ""),
            "phone": raw_data.get("mobile_number", ""),
        },
        "education": process_education(raw_data.get("education", [])),
        "work_experience": process_experience(raw_data.get("experience", [])),
        "skills": {
            "technical_skills": raw_data.get("skills", []),
            "soft_skills": [],  # pyresparser doesn't differentiate between technical and soft skills
        },
        "total_experience": {
            "years": 0,
            "months": 0,
        }
    }
    
    # Calculate total experience
    years, months = calculate_total_experience(formatted_data["work_experience"])
    formatted_data["total_experience"]["years"] = years
    formatted_data["total_experience"]["months"] = months
    
    return formatted_data

def save_extracted_information(information, file_name, csv_path='exinfo_test.csv'):
    try:
        # Convert the information dictionary to a JSON string
        information_json = json.dumps(information, indent=2)

        # Create directory if it doesn't exist
        directory = os.path.dirname(csv_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Check if the CSV file already exists
        if os.path.exists(csv_path):
            existing_data = pd.read_csv(csv_path)
        else:
            existing_data = pd.DataFrame(columns=["FileName", "ExtractedInformation"])

        # Update the existing row or create a new row
        if file_name in existing_data['FileName'].values:
            existing_data.loc[existing_data['FileName'] == file_name, 'ExtractedInformation'] = information_json
        else:
            new_row = pd.DataFrame({"FileName": [file_name], "ExtractedInformation": [information_json]})
            existing_data = pd.concat([existing_data, new_row], ignore_index=True)

        # Save the updated data back to the CSV file
        existing_data.to_csv(csv_path, index=False)
        logger.info(f"Information saved successfully for file: {file_name}")
    except Exception as e:
        logger.error(f"Error saving information for file {file_name}: {str(e)}")

def process_cv(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.getbuffer())
            temp_file_path = temp_file.name

        raw_info = extract_resume_info(temp_file_path)
        if raw_info:
            formatted_info = format_extracted_info(raw_info)
            save_extracted_information(formatted_info, file.name)
            return formatted_info
        else:
            logger.warning(f"No information extracted from file: {file.name}")
            return None
    except Exception as e:
        logger.error(f"Error processing file {file.name}: {str(e)}")
        return None
    finally:
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)

def main():
    st.title('Comprehensive CV Information Extraction')

    uploaded_files = st.file_uploader("Upload CV(s) (PDF)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        if st.button('Process CV(s)'):
            for uploaded_file in uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    extracted_info = process_cv(uploaded_file)
                    if extracted_info:
                        st.subheader(f"Extracted Information for {uploaded_file.name}")
                        st.json(extracted_info)
                        st.success(f"Information extracted and saved successfully for {uploaded_file.name}!")
                    else:
                        st.error(f"Failed to extract information from {uploaded_file.name}")

if __name__ == "__main__":
    main()
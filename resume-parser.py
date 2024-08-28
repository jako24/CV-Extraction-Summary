import streamlit as st
import tempfile
import os
import json
import re
import logging
from datetime import datetime
from dateutil import parser
from dateutil.relativedelta import relativedelta
import pandas as pd
import pdfplumber
import spacy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except IOError:
    logger.info("Downloading spaCy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
        return None

def extract_name(text):
    doc = nlp(text[:1000])  # Process only the first 1000 characters for efficiency
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return ""

def extract_contact_info(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b(?:\+?1[-.]?)?\(?[2-9][0-9]{2}\)?[-.]?[2-9][0-9]{2}[-.]?[0-9]{4}\b'
    
    email = re.search(email_pattern, text)
    phone = re.search(phone_pattern, text)
    
    return {
        "email": email.group() if email else "",
        "phone": phone.group() if phone else ""
    }

def extract_education(text):
    education_keywords = ["education", "academic background", "qualification"]
    education_section = ""
    for keyword in education_keywords:
        match = re.search(f"{keyword}.*", text, re.IGNORECASE | re.DOTALL)
        if match:
            education_section = match.group()
            break
    
    if not education_section:
        return []
    
    education_list = []
    lines = education_section.split('\n')
    current_edu = {}
    for line in lines:
        if re.search(r'\b(Bachelor|Master|PhD|BSc|MSc|MBA)\b', line, re.IGNORECASE):
            if current_edu:
                education_list.append(current_edu)
            current_edu = {
                "degree": "",
                "branch": "",
                "institution": "",
                "graduation_date": ""
            }
            current_edu["degree"] = line.strip()
        elif current_edu:
            if not current_edu["institution"]:
                current_edu["institution"] = line.strip()
            elif not current_edu["graduation_date"]:
                date_match = re.search(r'\b\d{4}\b', line)
                if date_match:
                    current_edu["graduation_date"] = date_match.group()
    
    if current_edu:
        education_list.append(current_edu)
    
    return education_list

def extract_work_experience(text):
    experience_keywords = ["work experience", "professional experience", "employment history"]
    experience_section = ""
    for keyword in experience_keywords:
        match = re.search(f"{keyword}.*", text, re.IGNORECASE | re.DOTALL)
        if match:
            experience_section = match.group()
            break
    
    if not experience_section:
        return []
    
    experience_list = []
    lines = experience_section.split('\n')
    current_job = {}
    for line in lines:
        if re.search(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}\b', line, re.IGNORECASE):
            if current_job:
                experience_list.append(current_job)
            current_job = {
                "company_name": "",
                "job_title": "",
                "start_date": "",
                "end_date": ""
            }
            dates = re.findall(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}\b', line, re.IGNORECASE)
            if len(dates) >= 2:
                current_job["start_date"] = dates[0]
                current_job["end_date"] = dates[1]
            elif len(dates) == 1:
                current_job["start_date"] = dates[0]
                current_job["end_date"] = "Present"
        elif current_job:
            if not current_job["job_title"]:
                current_job["job_title"] = line.strip()
            elif not current_job["company_name"]:
                current_job["company_name"] = line.strip()
    
    if current_job:
        experience_list.append(current_job)
    
    return experience_list

def extract_skills(text):
    skills_keywords = ["skills", "technical skills", "core competencies"]
    skills_section = ""
    for keyword in skills_keywords:
        match = re.search(f"{keyword}.*", text, re.IGNORECASE | re.DOTALL)
        if match:
            skills_section = match.group()
            break
    
    if not skills_section:
        return {"technical_skills": [], "soft_skills": []}
    
    skills = re.findall(r'\b[A-Za-z]+(?:\s+[A-Za-z]+)*\b', skills_section)
    # This is a simple split, you might want to implement a more sophisticated
    # method to differentiate between technical and soft skills
    return {
        "technical_skills": skills[:len(skills)//2],
        "soft_skills": skills[len(skills)//2:]
    }

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

def format_extracted_info(text):
    formatted_data = {
        "name": extract_name(text),
        "contact_info": extract_contact_info(text),
        "education": extract_education(text),
        "work_experience": extract_work_experience(text),
        "skills": extract_skills(text),
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

        text = extract_text_from_pdf(temp_file_path)
        if text:
            formatted_info = format_extracted_info(text)
            save_extracted_information(formatted_info, file.name)
            return formatted_info
        else:
            logger.warning(f"No text extracted from file: {file.name}")
            return None
    except Exception as e:
        logger.error(f"Error processing file {file.name}: {str(e)}")
        return None
    finally:
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)

def main():
    st.title('Custom CV Information Extraction')

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
import streamlit as st
from resume_parser import resumeparse
import tempfile
import os
from datetime import datetime
import pandas as pd
import json
import logging
import spacy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except IOError:
        st.error("Failed to load the spaCy model. Please make sure it's installed correctly.")
        return None

nlp = load_spacy_model()

def parse_resume(file_path):
    try:
        data = resumeparse.read_file(file_path)
        return data
    except Exception as e:
        logger.error(f"Error parsing resume: {str(e)}")
        return None

def save_extracted_information(information, method, file_name, csv_path='exinfo_test.csv'):
    columns = ["FileName", "ResumeParserExtraction", "OCRExtraction", "CorrectExtraction"]

    information = {k.lower(): v.lower() if isinstance(v, str) else v for k, v in information.items()}
    information_json = json.dumps(information, indent=2).lower()

    directory = os.path.dirname(csv_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    if os.path.exists(csv_path):
        existing_data = pd.read_csv(csv_path)
        for col in columns:
            if col not in existing_data.columns:
                existing_data[col] = ''
    else:
        existing_data = pd.DataFrame(columns=columns)

    if file_name in existing_data['FileName'].values:
        existing_data.loc[existing_data['FileName'] == file_name, method] = information_json
    else:
        new_row = pd.DataFrame([{col: '' for col in columns}])
        new_row['FileName'] = file_name
        new_row[method] = information_json
        existing_data = pd.concat([existing_data, new_row], ignore_index=True)

    existing_data.to_csv(csv_path, index=False)

def process_resume(uploaded_file):
    try:
        logger.info("Starting resume processing...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name

        logger.info("Parsing resume with resume-parser...")
        parsed_data = parse_resume(temp_file_path)
        
        if parsed_data:
            logger.info("Resume parsing completed successfully")
            save_extracted_information(parsed_data, 'ResumeParserExtraction', uploaded_file.name)
            return parsed_data
        else:
            logger.error("Resume parsing failed")
            return None
    except Exception as e:
        logger.error(f"An error occurred while processing the resume: {str(e)}")
        st.error(f"An error occurred while processing the resume: {str(e)}")
        return None

st.title('Enhanced Resume Information Extraction')
uploaded_file = st.file_uploader(label='Upload Resume', type=['pdf', 'docx', 'doc'])

if 'parsed_data' not in st.session_state:
    st.session_state.parsed_data = None

if uploaded_file:
    if st.button(label='Process Resume'):
        with st.spinner("Processing resume..."):
            st.session_state.parsed_data = process_resume(uploaded_file)

    if st.session_state.parsed_data:
        st.subheader("Extracted Information")
        st.json(st.session_state.parsed_data)
    elif st.session_state.parsed_data is not None:
        st.error("Resume parsing failed. Please try again with a different file.")

# Debug information
st.write("Debug Info:")
st.write(f"Uploaded file: {uploaded_file}")
if uploaded_file:
    st.write(f"File type: {uploaded_file.type}")
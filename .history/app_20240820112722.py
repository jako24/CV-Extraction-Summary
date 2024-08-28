import re
import os
import json
import openai
import pytesseract
from pdf2image import convert_from_path
import cv2
import tempfile
import json
from dateutil import parser
from dateutil.relativedelta import relativedelta
import easyocr
import requests
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import numpy

openai.api_key = os.getenv('OPENAI_API_KEY', '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')

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(extracted_text):
    # Remove dots and extra whitespace
    cleaned_text = re.sub(r'\s*\.\s*', ' ', extracted_text)
    return cleaned_text

def pdf_to_img(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_file.getbuffer())
        temp_file_path = temp_file.name

    # pdf_pages = convert_from_path(temp_file_path, dpi=500, poppler_path='/opt/homebrew/bin')
    pdf_pages = convert_from_path(temp_file_path, dpi=500)
    img_list = []
    for i, page in enumerate(pdf_pages, start=1):
        img_path = f'page_{i}.jpg'
        page.save(img_path, 'JPEG')
        img_list.append(img_path)
    print("PDF to Image Conversion Successful!")
    return img_list

def bounding_boxes(img_list, show_boxes):
    boxes = {}
    for curr_img in img_list:
        img = cv2.imread(curr_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilate = cv2.dilate(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(dilate, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        temp = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 5000:
                continue
            temp.append([x, y, w, h])
            if show_boxes:
                cv2.rectangle(img, (x, y), (x + w, y + h), color=(255, 0, 255), thickness=2)
        if show_boxes:
            img = cv2.resize(img, (500, 700), interpolation=cv2.INTER_AREA)
            st.image(image=img, caption=curr_img)
        boxes[curr_img] = temp
    print(f'Contours saved Successfully! Number of images processed: {len(boxes)}')
    return boxes

# def extract_text(boxes):
#     pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
#     text = ''
#     for key in boxes:
#         img = cv2.imread(key)
#         if isinstance(boxes[key], list):
#             for box in boxes[key]:
#                 x, y, w, h = box
#                 cropped_image = img[y:y + h, x:x + w]
#                 gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#                 _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#                 text += str(pytesseract.image_to_string(thresh, config='--psm 6'))
#     print('Text Extraction Completed!')
#     return text

def extract_text_ocr(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_file.getbuffer())
        temp_file_path = temp_file.name

    pdf_pages = convert_from_path(temp_file_path, dpi=300)
    use_gpu = torch.cuda.is_available()
    reader = easyocr.Reader(['en'], gpu=use_gpu)
    text = ''

    for page in pdf_pages:
        result = reader.readtext(numpy.array(page))
        text += ' '.join([res[1] for res in result]) + '\n\n'

    return text

def find_gaps(section_densities, section_width, threshold=0.1):
    gaps = []
    for i in range(1, len(section_densities)):
        if section_densities[i] < threshold * max(section_densities):
            gaps.append(i * section_width)
    return gaps

def extract_text_dynamic_column(img_list):
    use_gpu = torch.cuda.is_available()
    reader = easyocr.Reader(['en'], gpu=use_gpu)
    text = ''

    for img_path in img_list:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not load image {img_path}")
            continue

        height, width, _ = img.shape
        num_sections = 20
        section_width = width // num_sections
        section_densities = []

        for i in range(num_sections):
            section_img = img[:, i * section_width:(i + 1) * section_width]
            gray = cv2.cvtColor(section_img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (9, 9), 0)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            density = cv2.countNonZero(thresh)
            section_densities.append(density)

        gaps = find_gaps(section_densities, section_width)

        if gaps:
            left_img = img[:, :gaps[0]]
            right_img = img[:, gaps[0]:]

            left_result = reader.readtext(left_img, detail=0)
            for extracted_text in left_result:
                cleaned_text = clean_text(extracted_text)
                text += cleaned_text + '\n\n'

            right_result = reader.readtext(right_img, detail=0)
            for extracted_text in right_result:
                cleaned_text = clean_text(extracted_text)
                text += cleaned_text + '\n\n'
        else:
            left_img = img[:, :width // 2]
            right_img = img[:, width // 2:]

            left_result = reader.readtext(left_img, detail=0)
            for extracted_text in left_result:
                cleaned_text = clean_text(extracted_text)
                text += cleaned_text + '\n\n'

            right_result = reader.readtext(right_img, detail=0)
            for extracted_text in right_result:
                cleaned_text = clean_text(extracted_text)
                text += cleaned_text + '\n\n'

    print(f'Dynamic Column Text Extraction Completed! Extracted text length: {len(text)}')
    return text

def extract_text_pypdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text = ''
    for doc in documents:
        extracted_text = doc.page_content
        cleaned_text = clean_text(extracted_text)
        text += cleaned_text + '\n\n'
    return text

def merge_information(ocr_info, pypdf_info):
    merged_info = ocr_info.copy()  # Start with OCR info as the base

    for key, value in pypdf_info.items():
        if key not in merged_info or not merged_info[key]:
            merged_info[key] = value
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if sub_key not in merged_info[key] or not merged_info[key][sub_key]:
                    merged_info[key][sub_key] = sub_value
        elif isinstance(value, list):
            merged_info[key] = list(set(merged_info[key] + value))

    return merged_info

def calculate_total_experience(work_experience):
    total_experience = relativedelta()
    current_terms = {'present', 'now', 'current', 'ongoing', 'till now'}

    for job in work_experience:
        try:
            start_date = parser.parse(job['start_date'])
            end_date_str = job['end_date'].lower()
            if end_date_str in current_terms:
                end_date = datetime.now()
            else:
                end_date = parser.parse(job['end_date'])
            total_experience += relativedelta(end_date, start_date)
        except Exception as e:
            print(f"Error parsing dates: {e}")

    return total_experience.years, total_experience.months

def gpt3_extract_information(text):
    url = 'http://10.82.213.205:9013/v1/completions'
    prompt = f"""Extract the following information from the text and format it as a JSON:
    {{
        "name": "",
        "contact_info": {{
            "email": "",
            "phone": "",
        }},
        "education": [
            {{
                "degree": "",
                "branch": "",
                "institution": "",
                "graduation_date": "",
            }}
        ],
        "work_experience": [
            {{
                "company_name": "",
                "job_title": "",
                "start_date": "",
                "end_date": "",
            }}
        ],
        "skills": {{
            "technical_skills": [],
            "soft_skills": [],
        }},
        "total_experience": {{
            "years": 0,
            "months": 0,
        }}
    }}
    Ensure that all extracted information is complete and accurate based on the provided text.
    Text: {text}
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an experienced recruiter, that is looking for the best employees to hire. Find all of the important information and fill it into the JSON."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7,
    )

    json_output = response['choices'][0]['message']['content'].strip()

    # Ensure the output is valid JSON
    try:
        # Fix common issues with the JSON format
        json_output = json_output.replace("```json", "").replace("```", "").strip()
        json_output = json_output.replace("\n", "").replace("\\", "")
        json_output = json.loads(json_output)

        # Calculate total experience
        years, months = calculate_total_experience(json_output['work_experience'])
        json_output['total_experience']['years'] = years
        json_output['total_experience']['months'] = months

    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return {}

    return json_output

def gpt3_generate_summary(information):
    prompt = f"""
    Based on the following candidate information, generate a concise summary and a recommendation for a recruiter:
    Information: {json.dumps(information, indent=2)}

    Summary:
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an experienced recruiter, creating a summary and recommendation for a candidate based on their information."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.7,
    )

    summary = response['choices'][0]['message']['content'].strip()
    return summary

def pypdf_loader(uploaded_file):
    if uploaded_file:
        file_name = uploaded_file.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name

        text = extract_text_pypdf(temp_file_path)
        information = gpt3_extract_information(text)
        summary = gpt3_generate_summary(information)

        # Save the extracted information to the CSV file
        save_extracted_information(information, method='PDFLoader', file_name=file_name)

        return text, information, summary
    else:
        st.warning('Please select a PDF file!')

def analyze_completeness(information):
    incomplete_fields = []
    for field, value in information.items():
        if not value or (isinstance(value, dict) and not any(value.values())):
            incomplete_fields.append(field)
        elif isinstance(value, list) and not value:
            incomplete_fields.append(field)
        elif isinstance(value, dict):
            for sub_field, sub_value in value.items():
                if not sub_value:
                    incomplete_fields.append(f"{field}.{sub_field}")
    return incomplete_fields

def enhance_with_ocr(pypdf_info, ocr_info, incomplete_fields):
    enhanced_info = pypdf_info.copy()
    for field in incomplete_fields:
        if '.' in field:  # Handling nested fields
            main_field, sub_field = field.split('.')
            if main_field in ocr_info and sub_field in ocr_info[main_field]:
                enhanced_info[main_field][sub_field] = ocr_info[main_field][sub_field]
        else:
            if field in ocr_info and ocr_info[field]:
                enhanced_info[field] = ocr_info[field]
    return enhanced_info

def save_extracted_information(information, method, file_name, csv_path='exinfo_test.csv'):
    # Define the columns based on the provided CSV structure
    columns = ["FileName", "PDFLoaderExtraction", "OCRExtraction", "CorrectExtraction"]

    # Convert the information dictionary to a JSON string and ensure all keys and values are lowercase
    information = {k.lower(): v.lower() if isinstance(v, str) else v for k, v in information.items()}
    information_json = json.dumps(information, indent=2).lower()

    # if it there is no file create it
    directory = os.path.dirname(csv_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Check if the CSV file already exists
    if os.path.exists(csv_path):
        existing_data = pd.read_csv(csv_path)
        if not 'FileName' in existing_data.columns:
            existing_data['FileName'] = ''
        if not 'PDFLoaderExtraction' in existing_data.columns:
            existing_data['PDFLoaderExtraction'] = ''
        if not 'OCRExtraction' in existing_data.columns:
            existing_data['OCRExtraction'] = ''
        if not 'CorrectExtraction' in existing_data.columns:
            existing_data['CorrectExtraction'] = ''
    else:
        existing_data = pd.DataFrame(columns=columns)
        existing_data["FileName"] = [file_name]
        existing_data["Attributes"] = ["ExtractedInformation"]

    # Update the existing row or create a new row based on the method
    if method == 'OCR':
        extraction_key = 'OCRExtraction'
    elif method == 'PDFLoader':
        extraction_key = 'PDFLoaderExtraction'
    elif method == 'Enhanced':
        extraction_key = 'EnhancedExtraction'
    else:
        raise ValueError(f"Unknown method: {method}")

    if extraction_key in existing_data.columns:
        if file_name in existing_data['FileName'].values:
            existing_data.loc[existing_data['FileName'] == file_name, extraction_key] = information_json
        else:
            new_row = {
                "FileName": file_name,
                "Attributes": "ExtractedInformation",
                "PDFLoaderExtraction": "",
                "OCRExtraction": "",
                "CorrectExtraction": "",
                "EnhancedExtraction": ""
            }
            new_row[extraction_key] = information_json
            new_row_df = pd.DataFrame([new_row])
            existing_data = pd.concat([existing_data, new_row_df], ignore_index=True)
    else:
        new_row = {
            "FileName": file_name,
            "Attributes": "ExtractedInformation",
            "PDFLoaderExtraction": "",
            "OCRExtraction": "",
            "CorrectExtraction": ""
        }
        new_row[extraction_key] = information_json
        new_row_df = pd.DataFrame([new_row])
        existing_data = pd.concat([existing_data, new_row_df], ignore_index=True)

    # Save the updated data back to the CSV file
    existing_data.to_csv(csv_path, index=False)

def ocr(uploaded_file, show_boxes):
    if not uploaded_file:
        logger.warning("No file uploaded for OCR")
        st.warning('Please select a PDF file!')
        return None, None, None

    try:
        file_name = uploaded_file.name
        logger.info(f"Starting OCR process for file: {file_name}")

        img_list = pdf_to_img(uploaded_file)
        logger.info(f"Converted PDF to {len(img_list)} images")

        if show_boxes:
            bounding_boxes(img_list, show_boxes)
            return None, None, None

        text = extract_text_dynamic_column(img_list)
        logger.info(f"Extracted text length: {len(text)}")

        information = gpt3_extract_information(text)
        logger.info("Information extracted using GPT-3")

        summary = gpt3_generate_summary(information)
        logger.info("Summary generated")

        save_extracted_information(information, method='OCR', file_name=file_name)
        logger.info(f"Saved extracted information for file: {file_name}")

        return text, information, summary
    except Exception as e:
        logger.error(f"Error in OCR process: {str(e)}", exc_info=True)
        st.error(f"An error occurred during OCR: {str(e)}")
        return None, None, None


def assign_confidence_score(extraction_method, field, value):
    base_score = 0.7 if extraction_method == 'OCR' else 0.8  # Assume PyPDF is slightly more reliable
    
    if field == 'education':
        # Higher confidence for well-structured education entries
        return base_score * (1 + 0.1 * sum(1 for k in ['degree', 'institution', 'graduation_date'] if k in value))
    elif field == 'skills':
        # Higher confidence for longer skill lists
        return base_score * min(1 + 0.01 * len(value), 1.5)
    else:
        return base_score

def add_confidence_scores(information, extraction_method):
    for field, value in information.items():
        if isinstance(value, list):
            for item in value:
                item['confidence'] = assign_confidence_score(extraction_method, field, item)
        elif isinstance(value, dict):
            for sub_field, sub_value in value.items():
                value[sub_field] = {
                    'value': sub_value,
                    'confidence': assign_confidence_score(extraction_method, sub_field, sub_value)
                }
        else:
            information[field] = {
                'value': value,
                'confidence': assign_confidence_score(extraction_method, field, value)
            }
    return information


def validate_merged_information(merged_info):
    def get_value(item):
        return item['value'] if isinstance(item, dict) and 'value' in item else item

    validated_info = merged_info.copy()
    
    # Validate education
    if 'education' in validated_info:
        validated_education = []
        for edu in validated_info['education']:
            if all(get_value(edu.get(k)) for k in ['degree', 'institution', 'graduation_date']):
                validated_education.append(edu)
            else:
                logger.warning(f"Incomplete education entry removed: {edu}")
        validated_info['education'] = validated_education

    # Validate work experience
    if 'work_experience' in validated_info:
        validated_work = []
        for work in validated_info['work_experience']:
            if all(get_value(work.get(k)) for k in ['company_name', 'job_title', 'start_date']):
                validated_work.append(work)
            else:
                logger.warning(f"Incomplete work experience entry removed: {work}")
        validated_info['work_experience'] = validated_work

    # Validate skills
    if 'skills' in validated_info:
        if isinstance(validated_info['skills'], dict):
            validated_info['skills']['technical_skills'] = list(set(get_value(skill) for skill in validated_info['skills'].get('technical_skills', [])))
            validated_info['skills']['soft_skills'] = list(set(get_value(skill) for skill in validated_info['skills'].get('soft_skills', [])))
        else:
            logger.warning(f"Invalid skills structure: {validated_info['skills']}")

    return validated_info

def merge_information(ocr_info, pypdf_info):
    def merge_lists(list1, list2):
        if all(isinstance(item, dict) for item in list1 + list2):
            # For lists of dictionaries, we can't use set() directly
            merged = list1 + list2
            # Remove duplicates based on some key, e.g., 'company_name' for work experience
            seen = set()
            unique_items = []
            for item in merged:
                key = tuple(item.items())  # Convert dict to hashable tuple
                if key not in seen:
                    seen.add(key)
                    unique_items.append(item)
            return unique_items
        else:
            return list(set(list1 + list2))

    def merge_education(ocr_edu, pypdf_edu):
        if all(isinstance(item, dict) and 'degree' in item for item in ocr_edu + pypdf_edu):
            merged_edu = ocr_edu + pypdf_edu
            return sorted(merged_edu, key=lambda x: x['degree'].get('confidence', 0) if isinstance(x['degree'], dict) else 0, reverse=True)
        else:
            return ocr_edu + pypdf_edu

    def merge_dict_values(v1, v2):
        if isinstance(v1, dict) and 'value' in v1 and isinstance(v2, dict) and 'value' in v2:
            return v1 if v1.get('confidence', 0) >= v2.get('confidence', 0) else v2
        else:
            return v1 or v2

    merged = {}

    all_keys = set(ocr_info.keys()) | set(pypdf_info.keys())
    for key in all_keys:
        if key not in ocr_info:
            merged[key] = pypdf_info[key]
        elif key not in pypdf_info:
            merged[key] = ocr_info[key]
        else:
            if isinstance(ocr_info[key], dict) and isinstance(pypdf_info[key], dict):
                if key == 'skills':
                    merged[key] = {
                        'technical_skills': merge_lists(ocr_info[key].get('technical_skills', []), pypdf_info[key].get('technical_skills', [])),
                        'soft_skills': merge_lists(ocr_info[key].get('soft_skills', []), pypdf_info[key].get('soft_skills', []))
                    }
                else:
                    merged[key] = {
                        subkey: merge_dict_values(ocr_info[key].get(subkey), pypdf_info[key].get(subkey))
                        for subkey in set(ocr_info[key].keys()) | set(pypdf_info[key].keys())
                    }
            elif isinstance(ocr_info[key], list) and isinstance(pypdf_info[key], list):
                if key == 'education':
                    merged[key] = merge_education(ocr_info[key], pypdf_info[key])
                elif key == 'work_experience':
                    merged[key] = merge_lists(ocr_info[key], pypdf_info[key])
                else:
                    merged[key] = ocr_info[key] + pypdf_info[key]
            else:
                merged[key] = merge_dict_values(ocr_info[key], pypdf_info[key])

    return merged

def process_resume(uploaded_file):
    try:
        logger.info("Starting resume processing...")
        
        # OCR extraction
        logger.info("Performing OCR extraction...")
        try:
            text_ocr = extract_text_ocr(uploaded_file)
            if not text_ocr:
                raise ValueError("OCR extraction returned empty text")
            logger.info("OCR text extraction completed")
            
            information_ocr = gpt3_extract_information(text_ocr)
            if not information_ocr:
                raise ValueError("GPT-3 information extraction from OCR text failed")
            information_ocr_with_confidence = add_confidence_scores(information_ocr, 'OCR')
            logger.info("OCR information extraction completed")
        except Exception as ocr_error:
            logger.error(f"Error in OCR extraction: {str(ocr_error)}")
            raise

        # PyPDFLoader extraction
        logger.info("Performing PyPDFLoader extraction...")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_file_path = temp_file.name
            logger.info(f"Temporary file created at: {temp_file_path}")
            
            text_pypdf = extract_text_pypdf(temp_file_path)
            if not text_pypdf:
                raise ValueError("PyPDFLoader extraction returned empty text")
            logger.info("PyPDFLoader text extraction completed")
            
            information_pypdf = gpt3_extract_information(text_pypdf)
            if not information_pypdf:
                raise ValueError("GPT-3 information extraction from PyPDF text failed")
            information_pypdf_with_confidence = add_confidence_scores(information_pypdf, 'PyPDF')
            logger.info("PyPDFLoader information extraction completed")
        except Exception as pypdf_error:
            logger.error(f"Error in PyPDFLoader extraction: {str(pypdf_error)}")
            raise

        # Merge information
        logger.info("Merging information...")
        try:
            merged_information = merge_information(information_ocr_with_confidence, information_pypdf_with_confidence)
            validated_information = validate_merged_information(merged_information)
            if not validated_information:
                raise ValueError("Information merging and validation failed")
            logger.info("Information merging and validation completed")
        except Exception as merge_error:
            logger.error(f"Error in merging information: {str(merge_error)}")
            raise

        # Generate summary
        logger.info("Generating final summary...")
        try:
            final_summary = gpt3_generate_summary(validated_information)
            if not final_summary:
                raise ValueError("Summary generation failed")
            logger.info("Summary generation completed")
        except Exception as summary_error:
            logger.error(f"Error in generating summary: {str(summary_error)}")
            raise

        # Combine OCR and PyPDF texts
        merged_text = f"OCR Extracted Text:\n{text_ocr}\n\nPyPDFLoader Extracted Text:\n{text_pypdf}"

        logger.info("Resume processing completed successfully")
        return merged_text, validated_information, final_summary

    except Exception as e:
        logger.error(f"An error occurred while processing the resume: {str(e)}", exc_info=True)
        st.error(f"An error occurred while processing the resume: {str(e)}")
        return None, None, None

st.title('Enhanced Resume Information Extraction')
uploaded_file = st.file_uploader(label='Upload PDF', type='pdf')

if 'text_ocr' not in st.session_state:
    st.session_state.text_ocr = ''
    st.session_state.information_ocr = {}

if 'text_pypdf' not in st.session_state:
    st.session_state.text_pypdf = ''
    st.session_state.information_pypdf = {}

if 'merged_text' not in st.session_state:
    st.session_state.merged_text = ''

if 'enhanced_info' not in st.session_state:
    st.session_state.enhanced_info = {}

if 'final_summary' not in st.session_state:
    st.session_state.final_summary = ''

if 'flag_ocr' not in st.session_state:
    st.session_state.flag_ocr = False

if 'flag_pypdf' not in st.session_state:
    st.session_state.flag_pypdf = False

if 'flag_processed' not in st.session_state:
    st.session_state.flag_processed = False

if uploaded_file:
    row0col1, row0col2, row0col3 = st.columns([1, 1, 1])

    with row0col1:
        if st.button(label='Run OCR'):
            with st.spinner("Please wait..."):
                st.session_state.text_ocr, st.session_state.information_ocr, st.session_state.summary_ocr = ocr(uploaded_file, show_boxes=False)
                if st.session_state.text_ocr:
                    st.session_state.flag_ocr = True
                else:
                    st.error("OCR extraction failed.")

        if st.session_state.flag_ocr:
            st.text_area("OCR Extracted Text", st.session_state.text_ocr, height=300)
            st.write("OCR Extracted Information:")
            st.json(st.session_state.information_ocr)

    with row0col2:
        if st.button(label='Run PyPDFLoader'):
            with st.spinner("Please wait..."):
                st.session_state.text_pypdf, st.session_state.information_pypdf, st.session_state.summary_pypdf = pypdf_loader(uploaded_file)
                if st.session_state.text_pypdf:
                    st.session_state.flag_pypdf = True
                else:
                    st.error("PyPDFLoader extraction failed.")

        if st.session_state.flag_pypdf:
            st.text_area("PyPDFLoader Extracted Text", st.session_state.text_pypdf, height=300)
            st.write("PyPDFLoader Extracted Information:")
            st.json(st.session_state.information_pypdf)

    with row0col3:
        if st.button(label='Process Resume'):
            with st.spinner("Processing resume..."):
                st.session_state.merged_text, st.session_state.enhanced_info, st.session_state.final_summary = process_resume(uploaded_file)
                if st.session_state.merged_text:
                    st.session_state.flag_processed = True
                else:
                    st.error("Resume processing failed.")

        if st.session_state.flag_processed:
            st.subheader("Extracted Text")
            st.text_area("Combined Extracted Text", st.session_state.merged_text, height=300)

            st.subheader("Enhanced Information")
            st.json(st.session_state.enhanced_info)

            # st.subheader("Final Summary")
            # st.write(st.session_state.final_summary)

# 35.198.246.197

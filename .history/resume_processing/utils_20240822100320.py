import re
import os
import json
import json
import pandas as pd
import logging
from ocr_processing import * 
from gpt_processing import * 
from pdf_processing import * 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def clean_text(extracted_text):
    # Remove dots and extra whitespace
    cleaned_text = re.sub(r'\s*\.\s*', ' ', extracted_text)
    return cleaned_text


csv_path='/Users/janekkorczynski/CVExtraction/test_cv_extraction - extracted_information_test_cv copy1 copy.csv'
def save_extracted_information(information, method, file_name, csv_path=csv_path):
    try:
        # Define the columns based on the provided CSV structure
        columns = ["FileName", "PDFLoaderExtraction", "OCRExtraction", "CombinedOCRPDFExtraction", "EasyOCRExtraction", "CorrectExtraction"]

        # Convert the information dictionary to a JSON string
        information_json = json.dumps(information, indent=2).lower()

        # Log the start of the saving process
        logger.info(f"Saving extracted information for {file_name} with method {method}.")

        # Ensure the directory exists
        directory = os.path.dirname(csv_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Check if the CSV file already exists
        if os.path.exists(csv_path):
            logger.info(f"CSV file found at {csv_path}. Reading existing data.")
            existing_data = pd.read_csv(csv_path)
        else:
            logger.info(f"CSV file does not exist. Creating new DataFrame with columns: {columns}.")
            existing_data = pd.DataFrame(columns=columns)

        # Ensure all necessary columns exist in the DataFrame
        for column in columns:
            if column not in existing_data.columns:
                logger.info(f"Adding missing column: {column}")
                existing_data[column] = ""

        # Determine the extraction key based on the method
        extraction_key = {
            'OCR': 'OCRExtraction',
            'PDFLoader': 'PDFLoaderExtraction',
            'Combined': 'CombinedOCRPDFExtraction',
            'EasyOCR': 'EasyOCRExtraction'
        }.get(method, '')

        if not extraction_key:
            raise ValueError(f"Unknown method: {method}")

        # Check if the file name already exists in the DataFrame
        if file_name in existing_data['FileName'].values:
            logger.info(f"File {file_name} found in CSV. Updating existing row.")
            existing_data.loc[existing_data['FileName'] == file_name, extraction_key] = information_json
        else:
            logger.info(f"File {file_name} not found in CSV. Adding new row.")
            new_row = pd.Series({
                "FileName": file_name,
                "PDFLoaderExtraction": "",
                "OCRExtraction": "",
                "CombinedOCRPDFExtraction": "",
                "EasyOCRExtraction": "",
                "CorrectExtraction": ""
            })
            new_row[extraction_key] = information_json
            existing_data = existing_data.append(new_row, ignore_index=True)

        # Save the updated data back to the CSV file
        existing_data.to_csv(csv_path, index=False)
        logger.info(f"Information for {file_name} saved successfully.")

    except Exception as e:
        logger.error(f"Failed to save information for {file_name}: {str(e)}")
        raise e


def process_resume(uploaded_file):
    try:
        logger.info("Starting resume processing...")
        
        # Capture the original file name
        file_name = uploaded_file.name

        # Step 1: OCR extraction
        logger.info("Performing OCR extraction...")
        try:
            # text_ocr, information_ocr, summary_ocr = ocr(uploaded_file, show_boxes=False)
            text_ocr = extract_text_ocr(uploaded_file, file_name, show_boxes=False)
            if not text_ocr:
                raise ValueError("OCR extraction returned empty text")
            logger.info("OCR extraction completed")
        except Exception as ocr_error:
            logger.error(f"Error in OCR extraction: {str(ocr_error)}")
            raise

        # Step 2: PyPDFLoader extraction for structured data
        logger.info("Performing PyPDFLoader extraction...")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_file_path = temp_file.name

            # Call pypdf_loader with the file path instead of the file object
            text_pypdf, information_pypdf, _ = pypdf_loader_merge(temp_file_path)
            # save_extracted_information(information_pypdf, method='EasyOCR', file_name=file_name)
            
            if not text_pypdf:
                raise ValueError("PyPDFLoader extraction returned empty text")
            logger.info("PyPDFLoader extraction completed")
        except Exception as pypdf_error:
            logger.error(f"Error in PyPDFLoader extraction: {str(pypdf_error)}")
            raise
        
        information_ocr = gpt3_extract_information(text_ocr)
        
        # Step 3: Merge information
        logger.info("Merging information from OCR and PyPDFLoader...")
        try:
            merged_information = merge_information(information_ocr, information_pypdf)
            save_extracted_information(merged_information, method='Combined', file_name=file_name)
            logger.info(f"Merged Information: {json.dumps(merged_information, indent=2)}")
        except Exception as merge_error:
            logger.error(f"Error in merging information: {str(merge_error)}")
            raise

        # Step 4: Generate summary
        logger.info("Generating final summary...")
        try:
            final_summary = gpt3_generate_summary(merged_information)
            if not final_summary:
                raise ValueError("Summary generation failed")
            logger.info("Summary generation completed")
        except Exception as summary_error:
            logger.error(f"Error in generating summary: {str(summary_error)}")
            raise

        merged_text = f"OCR Extracted Text:\n{text_ocr}\n\nPyPDFLoader Extracted Text:\n{text_pypdf}"

        logger.info("Resume processing completed successfully")
        return merged_text, merged_information, final_summary

    except Exception as e:
        logger.error(f"An error occurred while processing the resume: {str(e)}", exc_info=True)
        st.error(f"An error occurred while processing the resume: {str(e)}")
        return None, None, None
import re
import os
import json
import json
import pandas as pd
import logging
from gpt_processing import * 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            # existing_data = existing_data.append(new_row, ignore_index=True)
            existing_data = pd.concat([existing_data, pd.DataFrame([new_row])], ignore_index=True)

        # Save the updated data back to the CSV file
        existing_data.to_csv(csv_path, index=False)
        logger.info(f"Information for {file_name} saved successfully.")

    except Exception as e:
        logger.error(f"Failed to save information for {file_name}: {str(e)}")
        raise e


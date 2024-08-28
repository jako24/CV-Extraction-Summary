import pytesseract
from pdf2image import convert_from_path
import cv2
import tempfile
import json
import easyocr
import torch
from utils import clean_text
import streamlit as st
import logging
from gpt_processing import *
from utils import save_extracted_information

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def extract_text_ocr(pdf_file, file_name, show_boxes=False):
    # Save the uploaded PDF file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_file.getbuffer())
        temp_file_path = temp_file.name

    # Convert the PDF to images (one image per page)
    pdf_pages = convert_from_path(temp_file_path, dpi=300)
    use_gpu = torch.cuda.is_available()
    reader = easyocr.Reader(['en'], gpu=use_gpu)

    text = ''
    img_list = []

    # Convert pages to images and store paths for bounding_boxes function
    for i, page in enumerate(pdf_pages, start=1):
        img_path = f'{file_name}_page_{i}.jpg'
        page.save(img_path, 'JPEG')
        img_list.append(img_path)

    # Get bounding boxes for each page
    boxes = bounding_boxes(img_list, show_boxes)

    for img_path in img_list:
        img = cv2.imread(img_path)

        # Check if bounding boxes are detected for the image
        if img_path in boxes:
            for (x, y, w, h) in boxes[img_path]:
                # Crop the region of interest (ROI) based on bounding box
                roi = img[y:y+h, x:x+w]

                # Preprocess the ROI for better OCR accuracy
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                denoised = cv2.fastNlMeansDenoising(gray, h=10)
                _, binary_img = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Use EasyOCR to read text from the preprocessed ROI
                result = reader.readtext(binary_img, detail=0, paragraph=True)

                # Join the recognized text into a single string
                page_text = ' '.join(result)

                # Clean up the extracted text by removing unwanted extra spaces and formatting
                page_text = ' '.join(page_text.split())

                # Append to the final text result
                text += page_text + '\n\n'

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
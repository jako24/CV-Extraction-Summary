from langchain_community.document_loaders import PyPDFLoader
from utils import clean_text

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
    def merge_skills(ocr_skills, pypdf_skills):
        # Merge technical and soft skills, ensuring no duplicates
        technical_skills = list(set(ocr_skills.get("technical_skills", []) + pypdf_skills.get("technical_skills", [])))
        soft_skills = list(set(ocr_skills.get("soft_skills", []) + pypdf_skills.get("soft_skills", [])))
        return {
            "technical_skills": technical_skills,
            "soft_skills": soft_skills
        }

    # Start with OCR info as the base
    merged = ocr_info.copy()

    # Merge 'name': If not present in PyPDFLoader, use OCR extracted name
    merged['name'] = pypdf_info.get('name') or merged.get('name', '')

    # Merge contact info: prioritize PyPDFLoader, but fall back to OCR if PyPDF is missing data
    merged['contact_info'] = merged.get('contact_info', {})
    pypdf_contact = pypdf_info.get('contact_info', {})
    
    # Ensure email and phone are taken from the better source or fallback to OCR if necessary
    merged['contact_info']['email'] = pypdf_contact.get('email') or merged['contact_info'].get('email', '')
    merged['contact_info']['phone'] = pypdf_contact.get('phone') or merged['contact_info'].get('phone', '')

    # Merge education: Use PyPDFLoader's data if it provides more information, otherwise fallback on OCR
    if 'education' in pypdf_info and (not merged.get('education') or len(merged.get('education')) == 0):
        merged['education'] = pypdf_info['education']

    # Merge work experience: Use PyPDFLoader's data if OCR data is missing or incomplete
    if 'work_experience' in pypdf_info and (not merged.get('work_experience') or len(merged.get('work_experience')) == 0):
        merged['work_experience'] = pypdf_info['work_experience']

    # Merge skills from both sources
    if 'skills' in pypdf_info and 'skills' in merged:
        merged['skills'] = merge_skills(merged['skills'], pypdf_info['skills'])
    elif 'skills' in pypdf_info:
        merged['skills'] = pypdf_info['skills']

    return merged
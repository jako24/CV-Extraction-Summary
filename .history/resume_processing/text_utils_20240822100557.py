import re



def clean_text(extracted_text):
    # Remove dots and extra whitespace
    cleaned_text = re.sub(r'\s*\.\s*', ' ', extracted_text)
    return cleaned_text

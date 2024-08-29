# Enhanced Resume Information Extraction and Comparison

## Project Overview

This project automates the extraction of relevant information from resumes and compares candidates against specific job descriptions. By leveraging Optical Character Recognition (OCR), PDF parsing, and AI-generated summaries, the system ranks candidates and provides detailed evaluations. It is designed to streamline the recruitment process, making it easier to identify the best candidates for a given job role.

## Features

- **Resume Upload & Text Extraction:** Supports both structured and unstructured resumes, extracting relevant details using OCR (for scanned or image-based resumes) and PDF parsing (for digitally generated PDFs).
- **Job Description Comparison:** Compares extracted resume data with job descriptions and calculates scores based on the candidate’s fit for the role.
- **AI-Powered Summaries:** Utilizes GPT-3 to generate candidate summaries and recommendations, providing insights into each candidate’s strengths and areas for improvement.
- **Detailed Evaluation Output:** Displays extracted information, assigned scores, and AI-generated summaries directly in the web interface. Results are also stored in JSON format for future reference.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/resume-extraction-comparison.git
   cd resume_processig
   ```

2. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Streamlit web application:**

   ```bash
   streamlit run main.py
   ```

   This will launch the application on your local machine at [http://localhost:8501](http://localhost:8501).

2. **Enable remote access (optional):**

   To allow others to access the app remotely, you can use LocalTunnel:

   ```bash
   lt --port 8501
   ```

   This command will generate a public URL that you can share for remote access.

### Interaction with the System

1. **Input Job Description:**
   - Input a job description URL (e.g., from Infosys Careers).
   - The system scrapes the job description and extracts the necessary information.

2. **Upload Candidate's Resume:**
   - Upload the candidate’s resume in PDF format.
   - The system processes the resume, extracts relevant details (e.g., name, contact info, education, work experience, skills), and compares them with the job description.

3. **View and Compare Results:**
   - The system displays the extracted information, the candidate’s score, and the AI-generated summary.
   - You can compare multiple candidates against the same job description.

4. **Revisit Candidates:**
   - The system allows you to review or modify a different candidate's information at any time, which is useful for comparing multiple candidates.

## File Descriptions

- **`main.py`:** The entry point for the Streamlit web application, handling the user interface and orchestrating the text extraction and comparison processes.
- **`ocr_processing.py`:** Handles OCR using EasyOCR, particularly useful for unstructured or scanned resumes.
- **`pdf_processing.py`:** Extracts structured data from PDFs using PyPDFLoader, ideal for well-formatted digital resumes.
- **`job_comparison.py`:** Implements the logic for comparing resume data with job descriptions, using a weighted scoring system.
- **`gpt_processing.py`:** Utilizes OpenAI’s GPT-3 to generate candidate summaries and recommendations based on the comparison results.
- **`utils.py` and `text_utils.py`:** Contains various utility functions for text cleaning, date parsing, and formatting to prepare the data for comparison.
- **`requirements.txt`:** Lists all the Python dependencies required to run the project.
- **`comparison_results.json`:** Stores the results of the job comparison process, including candidate names, scores, and AI-generated summaries.

## Accuracy Evaluation

The project includes several methods to evaluate the effectiveness of text extraction from resumes:

1. **PyTesseract (OCR):** Best for graphical PDFs but may be affected by image quality and layout complexity.
2. **EasyOCR:** Advanced OCR tool that supports multiple languages and performs well across various fonts and layouts.
3. **PyPDFLoader:** Reliable for extracting structured data from digital PDFs, though it may struggle with scanned documents.
4. **Combination of EasyOCR and PyPDFLoader:** Merges the strengths of both tools, improving accuracy across a wide range of resume formats.

## Why the Combination is Better

The combined use of EasyOCR and PyPDFLoader provides a robust solution for resume extraction:

- **Strengths of Both Methods:** PyPDFLoader excels at structured data extraction, while EasyOCR handles complex or unstructured documents.
- **Improved Accuracy:** Combining these methods enhances overall accuracy by leveraging their respective strengths.
- **Comprehensive Extraction:** This approach captures more detailed information and reduces the likelihood of errors.

## Choosing the Right AI Model

Consider a combination of models for optimal performance:

- **GPT-3.5 Turbo:** For general-purpose extraction and summarization.
- **LLaMA 3.1:** For detailed reasoning and specialized knowledge extraction.
- **Gemma:** For generating professional and high-quality summaries.

For more information, visit [artificialanalysis.ai/models](https://artificialanalysis.ai/models).

## Conclusion

This project provides an advanced system for extracting and comparing resume data with job descriptions, making it an invaluable tool for recruiters. By combining different text extraction techniques and leveraging AI for summary generation, it ensures that recruiters can make informed decisions based on comprehensive candidate evaluations.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

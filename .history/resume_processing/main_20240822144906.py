import streamlit as st
from ocr_processing import extract_text_ocr, process_resume
from gpt_processing import gpt3_extract_information, gpt3_generate_summary, gpt3_extract_job_information
from utils import save_extracted_information
import subprocess
import time
import plotly.express as px
import pandas as pd 
import os
from job_comparison import scrape_job_description, save_comparison_results


# ngrok maybe will be a better option here ??
def start_localtunnel(port):
    try: 
        process = subprocess.Popen(['lt', '--port', str(port)], stdout=subprocess.PIPE)
        time.sleep(2)
        for line in process.stdout:
            if b'kttps://' in line:
                public_url = line.decode('utf-8').strip()
                st.write(f'Your public URL is: {public_url}')
                print(f'Your public URL is: {public_url}')
                break
    except Exception as e: 
        st.error(f"An error occurred while starting localtunnel: {e}")
        print(f"An error occurred while starting localtunnel: {e}")                 


def display_job_description(job_data):
    """
    Display the job description stored in session state.
    """
    st.write(f"**Job Title**: {job_data['job_title']}")
    st.write(f"**Job Description**: {job_data['job_description']}")
    st.write(f"**Technical Skills**: {', '.join(job_data['required_skills']['technical_skills'])}")
    st.write(f"**Soft Skills**: {', '.join(job_data['required_skills']['soft_skills'])}")

def check_duplicate_name(candidates_data, candidate_name):
    """
    Check if a candidate with the same name already exists in the candidates_data list.
    """
    for candidate in candidates_data:
        if candidate['name'] == candidate_name:
            return True
    return False

# Function to create the interactive dashboard
def create_dashboard(candidates_data, job_description, file_path):
    # Get the top 10 candidates based on the score and save them
    comparison_results = save_comparison_results(candidates_data, job_description, file_path)
    
    # Create a dataframe for the top 10 candidates
    df = pd.DataFrame(comparison_results[:10])  # Only take the top 10

    # Create a bar chart using Plotly
    fig = px.bar(df, x='name', y='score', title="Top 10 Candidates", 
                 hover_data=['summary'], labels={'score':'Score', 'name':'Candidate Name'})
    
    # Display the dashboard with the chart and candidate details
    st.title("Candidate Comparison Dashboard")
    
    st.plotly_chart(fig)
    
    # Allow recruiter to select a candidate for more details
    selected_candidate = st.selectbox("Select a Candidate for More Details", df['name'])
    
    if selected_candidate:
        candidate_summary = df[df['name'] == selected_candidate]['summary'].values[0]
        st.subheader(f"Details for {selected_candidate}")
        st.write(candidate_summary)

def main():
    st.title('Enhanced Resume Information Extraction')

    # URL input from recruiter
    url = st.text_input("Enter the job description URL:", "")

    # Display the current job description if it exists in session state
    if 'job_data' in st.session_state:
        st.subheader("Current Job Description Stored")
        display_job_description(st.session_state['job_data'])

    # Scrape and process job description if a new URL is entered
    if st.button("Scrape and Process Job Description"):
        if url:
            with st.spinner("Scraping job description..."):
                job_text = scrape_job_description(url)
                
                if "Error" in job_text:
                    st.error(f"Error: {job_text['Error']}")
                else:
                    with st.spinner("Processing job description with GPT..."):
                        job_data = gpt3_extract_job_information(job_text)
                        
                        if job_data:
                            # Store job data in session state
                            st.session_state['job_data'] = job_data
                            st.success("Job description successfully processed!")
                            display_job_description(job_data)
    
    # File uploader for PDF resumes
    uploaded_file = st.file_uploader(label='Upload PDF', type='pdf')
    
    # Initialize session state variables for candidate processing
    if 'candidates_data' not in st.session_state:
        st.session_state.candidates_data = []

    # Initialize session state variables
    if 'text_ocr' not in st.session_state:
        st.session_state.text_ocr = ''
        st.session_state.information_ocr = {}
        st.session_state.summary_ocr = ''
    if 'merged_text' not in st.session_state:
        st.session_state.merged_text = ''
        st.session_state.enhanced_info = {}
        st.session_state.final_summary = ''
    if 'flag_processed' not in st.session_state:
        st.session_state.flag_processed = False

    # Process resume and display the extracted information + summary
    if uploaded_file:
        file_name = uploaded_file.name
        row1col1, row1col2 = st.columns([1, 1])

        # Column 1: OCR + PyPDFLoader Processing
        with row1col1:
            if st.button(label='OCR + PDF'):
                with st.spinner("Processing resume with OCR and PyPDFLoader..."):
                    st.session_state.merged_text, st.session_state.enhanced_info, st.session_state.final_summary = process_resume(uploaded_file)
                    
                    if st.session_state.merged_text:
                        st.session_state.flag_processed = True
                        st.session_state.candidates_data.append(st.session_state.enhanced_info)
                        st.success(f"Candidate data added: {st.session_state.enhanced_info['name']}")
                    else:
                        st.error("Resume processing failed.")

            if st.session_state.flag_processed:
                st.subheader("Combined Extracted Text")
                st.text_area("Combined Extracted Text", st.session_state.merged_text, height=300)
                
                st.subheader("Enhanced Information")
                st.json(st.session_state.enhanced_info)

        job_description = st.session_state.get('job_data', {})
        
        # Column 2: Candidate Summary
        with row1col2:
            if st.session_state.flag_processed:
                st.subheader("Candidate Summary")
                st.session_state.final_summary = gpt3_generate_summary(st.session_state.enhanced_info, job_description)
                st.write(st.session_state.final_summary)

        # Automatically update the dashboard as new candidates are processed
    if st.session_state.candidates_data and 'job_data' in st.session_state:
        job_description = st.session_state['job_data']
        candidates_data = st.session_state['candidates_data']
        
        # Save and compare candidates
        file_path = os.path.join(os.getcwd(), 'comparison_results.json')
        create_dashboard(candidates_data, job_description, file_path)

if __name__ == '__main__':
    main()


# def main():
    
#     start_localtunnel(8501)
    
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
    
#     st.title('Enhanced Resume Information Extraction')
#     uploaded_file = st.file_uploader(label='Upload PDF', type='pdf')
    
#     # URL input from recruiter
#     url = st.text_input("Enter the job description URL:", "")
    
#     # Scrape and process job description
#     if st.button("Scrape and Process Job Description"):
#         if url:
#             with st.spinner("Scraping job description..."):
#                 job_text = scrape_job_description(url)
                
#                 if "error" in job_text:
#                     st.error(f"Error: {job_text['error']}")
#                 else:
#                     # Use GPT to extract structured information
#                     with st.spinner("Processing job description with GPT..."):
#                         job_data = gpt3_extract_job_information(job_text)
                        
#                         if job_data:
#                             # Save the job description to session state
#                             st.session_state['job_data'] = job_data
#                             st.success("Job description successfully processed!")

#                             # Display the job description
#                             st.write(f"**Job Title**: {job_data['job_title']}")
#                             st.write(f"**Job Description**: {job_data['job_description']}")
#                             st.write(f"**Technical Skills**: {', '.join(job_data['required_skills']['technical_skills'])}")
#                             st.write(f"**Soft Skills**: {', '.join(job_data['required_skills']['soft_skills'])}")
#                             st.write(f"**Experience Required**: {job_data['experience_required']}")
#                             st.write(f"**Education Required**: {job_data['education_required']}")

#     # Display saved job description if it exists
#     if 'job_data' in st.session_state:
#         st.subheader("Current Job Description Stored")
#         st.write(f"**Job Title**: {st.session_state['job_data']['job_title']}")
#         st.write(f"**Job Description**: {st.session_state['job_data']['job_description']}")
#         st.write(f"**Technical Skills**: {', '.join(st.session_state['job_data']['required_skills']['technical_skills'])}")
#         st.write(f"**Soft Skills**: {', '.join(st.session_state['job_data']['required_skills']['soft_skills'])}")
#         st.write(f"**Experience Required**: {st.session_state['job_data']['experience_required']}")
#         st.write(f"**Education Required**: {st.session_state['job_data']['education_required']}")
    

#     # Initialize session state variables
#     if 'text_ocr' not in st.session_state:
#         st.session_state.text_ocr = ''
#         st.session_state.information_ocr = {}
#         st.session_state.summary_ocr = ''

#     if 'text_pypdf' not in st.session_state:
#         st.session_state.text_pypdf = ''
#         st.session_state.information_pypdf = {}

#     if 'merged_text' not in st.session_state:
#         st.session_state.merged_text = ''
#         st.session_state.enhanced_info = {}
#         st.session_state.final_summary = ''

#     if 'flag_ocr' not in st.session_state:
#         st.session_state.flag_ocr = False

#     if 'flag_pypdf' not in st.session_state:
#         st.session_state.flag_pypdf = False

#     if 'flag_processed' not in st.session_state:
#         st.session_state.flag_processed = False

#     if uploaded_file:
#         file_name = uploaded_file.name
#         row0col2, row0col3 = st.columns([1, 1])

#         # Column 1: OCR Processing
#         # with row0col1:
#         #     if st.button(label='Run OCR'):
#         #         with st.spinner("Running OCR..."):
#         #             st.session_state.text_ocr = extract_text_ocr(uploaded_file, show_boxes=False)
#         #             st.session_state.information_ocr = gpt3_extract_information(st.session_state.text_ocr)
#         #             st.session_state.summary_ocr = gpt3_generate_summary(st.session_state.information_ocr)
                    
#         #             if st.session_state.text_ocr:
#         #                 st.session_state.flag_ocr = True
#         #             else:
#         #                 st.error("OCR extraction failed.")

#         #     if st.session_state.flag_ocr:
#         #         st.text_area("OCR Extracted Text", st.session_state.text_ocr, height=300)
#         #         st.write("OCR Extracted Information:")
#         #         st.json(st.session_state.information_ocr)
#         #         st.write("OCR Summary:")
#         #         st.write(st.session_state.summary_ocr)

#         # Column 2: OCR + PyPDFLoader Processing
#         with row0col2:
#             if st.button(label='OCR + PDF'):
#                 with st.spinner("Processing resume with OCR and PyPDFLoader..."):
#                     st.session_state.merged_text, st.session_state.enhanced_info, st.session_state.final_summary = process_resume(uploaded_file)
                    
#                     if st.session_state.merged_text:
#                         st.session_state.flag_processed = True
#                     else:
#                         st.error("Resume processing failed.")

#             if st.session_state.flag_processed:
#                 st.subheader("Combined Extracted Text")
#                 st.text_area("Combined Extracted Text", st.session_state.merged_text, height=300)
                
#                 st.subheader("Enhanced Information")
#                 st.json(st.session_state.enhanced_info)
                
#                 # st.subheader("Final Summary")
#                 # st.write(st.session_state.final_summary)

#         # Column 3: EasyOCR-Only Processing
#         with row0col3:
#             if st.button(label='EasyOCR'):
#                 with st.spinner("Running EasyOCR..."):
#                     st.session_state.text_ocr = extract_text_ocr(uploaded_file, file_name, show_boxes=False)
#                     st.session_state.information_ocr = gpt3_extract_information(st.session_state.text_ocr)
#                     st.session_state.summary_ocr = gpt3_generate_summary(st.session_state.information_ocr)
                    
#                     save_extracted_information(st.session_state.information_ocr, method='EasyOCR', file_name=file_name)
                    
#                     if st.session_state.text_ocr:
#                         st.session_state.flag_ocr = True
#                     else:
#                         st.error("EasyOCR extraction failed.")

#             if st.session_state.flag_ocr:
#                 st.text_area("EasyOCR Extracted Text", st.session_state.text_ocr, height=300)
#                 st.write("EasyOCR Extracted Information:")
#                 st.json(st.session_state.information_ocr)
#                 # st.write("EasyOCR Summary:")
#                 # st.write(st.session_state.summary_ocr)




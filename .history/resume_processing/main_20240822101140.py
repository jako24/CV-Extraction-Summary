import streamlit as st
from pdf_processing import pypdf_loader, pypdf_loader_merge
from ocr_processing import extract_text_ocr, process_resume
from gpt_processing import gpt3_extract_information, gpt3_generate_summary
from utils import save_extracted_information

def main():
    
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    st.title('Enhanced Resume Information Extraction')
    uploaded_file = st.file_uploader(label='Upload PDF', type='pdf')

    # Initialize session state variables
    if 'text_ocr' not in st.session_state:
        st.session_state.text_ocr = ''
        st.session_state.information_ocr = {}
        st.session_state.summary_ocr = ''

    if 'text_pypdf' not in st.session_state:
        st.session_state.text_pypdf = ''
        st.session_state.information_pypdf = {}

    if 'merged_text' not in st.session_state:
        st.session_state.merged_text = ''
        st.session_state.enhanced_info = {}
        st.session_state.final_summary = ''

    if 'flag_ocr' not in st.session_state:
        st.session_state.flag_ocr = False

    if 'flag_pypdf' not in st.session_state:
        st.session_state.flag_pypdf = False

    if 'flag_processed' not in st.session_state:
        st.session_state.flag_processed = False

    if uploaded_file:
        file_name = uploaded_file.name
        row0col2, row0col3 = st.columns([1, 1])

        # Column 1: OCR Processing
        # with row0col1:
        #     if st.button(label='Run OCR'):
        #         with st.spinner("Running OCR..."):
        #             st.session_state.text_ocr = extract_text_ocr(uploaded_file, show_boxes=False)
        #             st.session_state.information_ocr = gpt3_extract_information(st.session_state.text_ocr)
        #             st.session_state.summary_ocr = gpt3_generate_summary(st.session_state.information_ocr)
                    
        #             if st.session_state.text_ocr:
        #                 st.session_state.flag_ocr = True
        #             else:
        #                 st.error("OCR extraction failed.")

        #     if st.session_state.flag_ocr:
        #         st.text_area("OCR Extracted Text", st.session_state.text_ocr, height=300)
        #         st.write("OCR Extracted Information:")
        #         st.json(st.session_state.information_ocr)
        #         st.write("OCR Summary:")
        #         st.write(st.session_state.summary_ocr)

        # Column 2: OCR + PyPDFLoader Processing
        with row0col2:
            if st.button(label='OCR + PDF'):
                with st.spinner("Processing resume with OCR and PyPDFLoader..."):
                    st.session_state.merged_text, st.session_state.enhanced_info, st.session_state.final_summary = process_resume(uploaded_file)
                    
                    if st.session_state.merged_text:
                        st.session_state.flag_processed = True
                    else:
                        st.error("Resume processing failed.")

            if st.session_state.flag_processed:
                st.subheader("Combined Extracted Text")
                st.text_area("Combined Extracted Text", st.session_state.merged_text, height=300)
                
                st.subheader("Enhanced Information")
                st.json(st.session_state.enhanced_info)
                
                # st.subheader("Final Summary")
                # st.write(st.session_state.final_summary)

        # Column 3: EasyOCR-Only Processing
        with row0col3:
            if st.button(label='EasyOCR'):
                with st.spinner("Running EasyOCR..."):
                    st.session_state.text_ocr = extract_text_ocr(uploaded_file, file_name, show_boxes=False)
                    st.session_state.information_ocr = gpt3_extract_information(st.session_state.text_ocr)
                    st.session_state.summary_ocr = gpt3_generate_summary(st.session_state.information_ocr)
                    
                    save_extracted_information(st.session_state.information_ocr, method='EasyOCR', file_name=file_name)
                    
                    if st.session_state.text_ocr:
                        st.session_state.flag_ocr = True
                    else:
                        st.error("EasyOCR extraction failed.")

            if st.session_state.flag_ocr:
                st.text_area("EasyOCR Extracted Text", st.session_state.text_ocr, height=300)
                st.write("EasyOCR Extracted Information:")
                st.json(st.session_state.information_ocr)
                # st.write("EasyOCR Summary:")
                # st.write(st.session_state.summary_ocr)



if __name__ == '__main__':
    main()
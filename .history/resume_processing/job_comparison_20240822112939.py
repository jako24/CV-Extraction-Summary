from gpt_processing import gpt3_extract_job_information
import requests
from bs4 import BeautifulSoup

def scrape_job_description(url):
    try: 
        # Send a GET request to the webpage
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the job description content in the div with class 'editable-sections'
        main_content = soup.find('div', class_='editable-sections')
        
        if main_content:
            # Extract the text content from the div
            job_text = main_content.get_text(separator="\n", strip=True)
            return job_text
        else:
            return {"Error": "Could not find the job description content."}
    
    except Exception as e:
        return {"Error": str(e)}




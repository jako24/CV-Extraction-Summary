from gpt_processing import gpt3_extract_job_information
import requests
from bs4 import BeautifulSoup

def scrape_job_description(url):
    try: 
        response = requests.get(url)
        response.raise_for_status()                     # Check for HTTP errors
        
        # Parse the HTTP content 
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Example scraping logic: Just het all text in teh main content area 
        main_content = soup.find('div', class_='editable-sections')
        job_text = main_content.get_text(separator="\n", strip=True)
        
        return job_text
    
    except Exception as e:
        return {"Error": str(e)}




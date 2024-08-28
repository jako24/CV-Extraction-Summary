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
        main_content = soup.find('div', class_='main_content')
        job_text = main_content.get_text(separator="\n", strip=True)
        
        return job_text
    
    except Exception as e:
        return {"Error": str(e)}


def scrape_job_description(url):
    try:
        # Send a GET request to the webpage
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors

        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Example scraping logic: Just get all text in the main content area
        main_content = soup.find('div', class_='main-content')  # Adjust the class to match your site's structure
        job_text = main_content.get_text(separator="\n", strip=True)

        return job_text

    except Exception as e:
        return {"error": str(e)}

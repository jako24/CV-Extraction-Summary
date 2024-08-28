from gpt_processing import gpt3_extract_job_information
import requests
from bs4 import BeautifulSoup
import re 

# Remove occurrences of "S/" 
def clean_text(text):
    cleaned_text = re.sub(r'\b[Ss]/', '', text)
    return cleaned_text

def scrape_job_description(url):
    try: 
        # Set headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
        }
        
        # Send a GET request to the webpage
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Check for HTTP errors
        
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the job description content in the div with class 'editable-sections'
        main_content = soup.find('div', class_='editable-sections')
        
        if main_content:
            # Extract the text content from the div
            job_text = main_content.get_text(separator="\n", strip=True)
            
            # Clean up the text to replace "S/He" with "They"
            cleaned_job_text = clean_text(job_text)
            
            # Print the cleaned content for analysis
            print("Cleaned Job Description:")
            print(cleaned_job_text)
            
            return cleaned_job_text
        else:
            print("Error: Could not find the job description content.")
            return {"Error": "Could not find the job description content."}
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"Error": str(e)}
    
    
# # Example usage:
# url = "https://digitalcareers.infosys.com/global-careers/company-job/description/reqid/120632BR"
# job_text = scrape_job_description(url)

# # Print the scraped information for analysis
# print("\nScraped Information for Analysis:")
# print(job_text)



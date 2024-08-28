from gpt_processing import gpt3_extract_job_information
import requests
from bs4 import BeautifulSoup
import streamlit as st
import pandas as pd
from gpt_processing import gpt3_generate_summary
import json
import re 

def scrape_job_description(url):
    try: 
        
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
            
            # Print the scraped content for analysis
            print("Scraped Job Description:")
            print(job_text)
            
            return job_text
        else:
            print("Error: Could not find the job description content.")
            return {"Error": "Could not find the job description content."}
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"Error": str(e)}


def extract_experience_from_text(experience_text):
    # Extract the first number found in the experience text
    match = re.search(r'\d+', str(experience_text))
    if match:
        return int(match.group(0))  # Return the first found number as an integer
    else:
        return 0  # Default to 0 if no number is found

def compare_candidates(candidates_data, job_description):
    scored_candidates = []

    for candidate in candidates_data:
        score = 0
        
        # Ensure that the experience values are integers
        candidate_experience_years = int(candidate['total_experience'].get('years', 0))
        
        # Extract experience from the job description safely
        job_experience_required_text = job_description.get('experience_required', '')
        job_experience_required = extract_experience_from_text(job_experience_required_text)
        
        # Check technical skills
        matching_technical_skills = set(candidate['skills']['technical_skills']).intersection(set(job_description['required_skills']['technical_skills']))
        score += len(matching_technical_skills) * 2.5  # Slightly lower weight for technical skills
        
        # Check soft skills
        matching_soft_skills = set(candidate['skills']['soft_skills']).intersection(set(job_description['required_skills']['soft_skills']))
        score += len(matching_soft_skills) * 1.5  # Slightly lower weight for soft skills
        
        # Check experience
        experience_gap = candidate_experience_years - job_experience_required
        
        if experience_gap >= 0:
            # Smaller bonus for meeting or exceeding the experience requirement
            score += 3 + (experience_gap * 0.5)  # Reduced points for exceeding experience requirement
        else:
            # Deduct more points for falling short of experience requirements
            score += max(0, 3 + (experience_gap * 0.7))  # Increased penalty for missing experience

        # Check education
        education_match = any(edu['degree'].lower() == job_description.get('education_required', '').lower() for edu in candidate['education'])
        if education_match:
            score += 4  # Slightly reduced weight for education match
        
        # Ensure the score is between 0 and 10
        score = max(0, min(score, 10))

        # Append candidate score
        scored_candidates.append({
            "name": candidate['name'],
            "score": score,
            "summary": gpt3_generate_summary(candidate, job_description)  # Summary for each candidate
        })
    
    # Sort candidates by score in descending order
    scored_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    return scored_candidates




# Function to save candidates comparison to a file
def save_comparison_results(candidates_data, job_description, file_path):
    comparison_results = compare_candidates(candidates_data, job_description)

    # Save the comparison results to a JSON file for future reference
    with open(file_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Inform the user that the data has been saved
    st.success(f"Comparison results saved to {file_path}")
    
    return comparison_results


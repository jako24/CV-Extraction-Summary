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
        job_experience_required = extract_experience_from_text(job_description.get('experience_required', ''))
        
        # Stricter Skill Matching
        matching_technical_skills = set(candidate['skills']['technical_skills']).intersection(set(job_description['required_skills']['technical_skills']))
        matching_soft_skills = set(candidate['skills']['soft_skills']).intersection(set(job_description['required_skills']['soft_skills']))
        
        # Award points only if there's an exact match, heavier penalties for missing required skills
        if len(matching_technical_skills) == len(job_description['required_skills']['technical_skills']):
            score += len(matching_technical_skills) * 5  # Full score for exact matches
        else:
            # Heavier penalty for missing critical skills
            penalty = (len(job_description['required_skills']['technical_skills']) - len(matching_technical_skills)) * 2
            score -= penalty
        
        if len(matching_soft_skills) == len(job_description['required_skills']['soft_skills']):
            score += len(matching_soft_skills) * 3  # Award full points for soft skills match
        else:
            penalty = (len(job_description['required_skills']['soft_skills']) - len(matching_soft_skills)) * 1.5
            score -= penalty

        # Stricter Experience Requirement
        experience_gap = candidate_experience_years - job_experience_required
        if experience_gap >= 0:
            # Higher bonus for exceeding experience
            score += 5 + (experience_gap * 2)  # Reward more experience with more points
        else:
            # Harsher penalty for lack of experience
            score += max(0, 5 + experience_gap)  # Deduct points heavily for not meeting the experience requirement
        
        # Stricter Education Requirement
        education_match = any(edu['degree'].lower() == job_description.get('education_required', '').lower() for edu in candidate['education'])
        if education_match:
            score += 5  # Award full points for meeting education requirement
        else:
            # Penalize candidates who don't meet education requirement
            score -= 3  # Harsher penalty for not meeting education requirement

        # Append candidate score
        scored_candidates.append({
            "name": candidate['name'],
            "score": max(0, score),  # Ensure the score does not go below zero
            "summary": gpt3_generate_summary(candidate, job_description)
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


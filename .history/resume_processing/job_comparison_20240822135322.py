from gpt_processing import gpt3_extract_job_information
import requests
from bs4 import BeautifulSoup
import streamlit as st
import pandas as pd
import plotly.express as px
from gpt_processing import gpt3_generate_summary
import os

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


# Function to compare candidates and assign scores based on job description
def compare_candidates(candidates_data, job_description):
    scored_candidates = []

    for candidate in candidates_data:
        score = 0
        
        # Check technical skills
        matching_technical_skills = set(candidate['skills']['technical_skills']).intersection(set(job_description['required_skills']['technical_skills']))
        score += len(matching_technical_skills) * 3  # Give higher weight to technical skills
        
        # Check soft skills
        matching_soft_skills = set(candidate['skills']['soft_skills']).intersection(set(job_description['required_skills']['soft_skills']))
        score += len(matching_soft_skills) * 2
        
        # Check experience
        experience_gap = abs(candidate['total_experience']['years'] - job_description.get('experience_required', 0))
        if candidate['total_experience']['years'] >= job_description.get('experience_required', 0):
            score += 5  # Bonus points for meeting or exceeding experience requirement
        
        # Check education
        education_match = any(edu['degree'].lower() == job_description.get('education_required', '').lower() for edu in candidate['education'])
        if education_match:
            score += 5  # Higher weight for education match
        
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

# Example Usage: Running the dashboard with sample data and saving results
def run_comparison_dashboard():
    # Job description example
    job_description = {
        "required_skills": {
            "technical_skills": ["python", "sql", "data analysis"],
            "soft_skills": ["communication", "teamwork"]
        },
        "experience_required": 3,  # in years
        "education_required": "Bachelor's Degree"
    }

    # Example candidates data (from resume parsing)
    candidates_data = [
        {
            "name": "John Doe",
            "skills": {
                "technical_skills": ["python", "sql"],
                "soft_skills": ["communication", "leadership"]
            },
            "education": [{"degree": "Bachelor's Degree"}],
            "total_experience": {"years": 4, "months": 5}
        },
        {
            "name": "Jane Smith",
            "skills": {
                "technical_skills": ["data analysis", "sql"],
                "soft_skills": ["teamwork", "problem-solving"]
            },
            "education": [{"degree": "Bachelor's Degree"}],
            "total_experience": {"years": 2, "months": 8}
        },
        # Add more candidates here
    ]
    
    # File path to save comparison results
    file_path = os.path.join(os.getcwd(), 'comparison_results.json')
    
    # Run the dashboard and save the comparison results
    create_dashboard(candidates_data, job_description, file_path)

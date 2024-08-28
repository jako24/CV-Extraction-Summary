import openai
import json
from dateutil import parser
from dateutil.relativedelta import relativedelta
from datetime import datetime
import os 

openai.api_key = os.getenv('OPENAI_API_KEY', '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')

def calculate_total_experience(work_experience):
    total_experience = relativedelta()
    current_terms = {'present', 'now', 'current', 'ongoing', 'till now', 'till date'}

    for job in work_experience:
        try:
            start_date = parser.parse(job['start_date'])
            end_date_str = job['end_date'].lower()

            # Handle "Not specified" and similar cases
            if end_date_str in current_terms or end_date_str in ['not specified', '']:
                end_date = datetime.now()
            else:
                end_date = parser.parse(job['end_date'])

            total_experience += relativedelta(end_date, start_date)
        except (ValueError, TypeError) as e:
            print(f"Error parsing dates: {e}")

    return total_experience.years, total_experience.months

def gpt3_extract_information(text):
    # url = 'http://10.82.213.205:9013/v1/completions' url for llama 8B does not work 
    prompt = f"""Extract the following information from the text and format it as a JSON:
    {{
        "name": "",
        "contact_info": {{
            "email": "",
            "phone": "",
        }},
        "education": [
            {{
                "degree": "",
                "branch": "",
                "institution": "",
                "graduation_date": "",
            }}
        ],
        "work_experience": [
            {{
                "company_name": "",
                "job_title": "",
                "start_date": "",
                "end_date": "",
            }}
        ],
        "skills": {{
            "technical_skills": [],
            "soft_skills": [],
        }},
        "total_experience": {{
            "years": 0,
            "months": 0,
        }}
    }}
    Ensure that all extracted information is complete and accurate based on the provided text.
    Text: {text}
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an experienced recruiter, that is looking for the best employees to hire. Find all of the important information and fill it into the JSON."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7,
    )

    json_output = response['choices'][0]['message']['content'].strip()

    # Ensure the output is valid JSON
    try:
        # Fix common issues with the JSON format
        json_output = json_output.replace("```json", "").replace("```", "").strip()
        json_output = json_output.replace("\n", "").replace("\\", "")
        json_output = json.loads(json_output)

        # Calculate total experience
        years, months = calculate_total_experience(json_output['work_experience'])
        json_output['total_experience']['years'] = years
        json_output['total_experience']['months'] = months

    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return {}

    return json_output

import openai
import json

def gpt3_generate_summary(information, job_description):
    # Combine candidate information and job description into the prompt
    prompt = f"""
    Based on the following candidate information and job description, generate a concise but thorough summary and a recommendation for a recruiter.
    
    Candidate Information:
    {json.dumps(information, indent=2)}
    
    Job Description:
    {json.dumps(job_description, indent=2)}
    
    Please follow these steps:
    1. **Summarize the candidate's experience and skills**, including both technical and soft skills, as well as their educational background and work experience. Mention any standout qualifications.
    2. **Evaluate how well the candidate's qualifications align with the job description**, focusing on the required skills, experience, and education. Highlight any relevant matches or gaps.
    3. **Provide a clear recommendation** on whether the candidate is a good fit for the described role. If they are a good fit, explain why. If they are not, suggest a different role or career path that better suits their qualifications.
    4. **Provide insights into potential areas of improvement** for the candidate that could help them better align with similar roles in the future.

    Be detailed in your assessment and provide actionable insights that would be useful for a recruiter making a hiring decision.
    
    Summary:
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an experienced recruiter tasked with analyzing candidate suitability for a job role based on their information and the job description."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1500,  # Increased token limit for more detailed output
        n=1,
        stop=None,
        temperature=0.5,  # Reduced temperature for more consistent and structured output
    )

    summary = response['choices'][0]['message']['content'].strip()
    return summary



def gpt3_extract_job_information(job_text):
    prompt = f"""
    Extract the following information from the job description text and format it as a detailed and complete JSON:
    {{
        "job_title": "",
        "job_description": "",
        "required_skills": {{
            "technical_skills": [],
            "soft_skills": [],
        }},
        "experience_required": "",
        "education_required": ""
    }}

    Ensure that all extracted information is accurate and as complete as possible. If certain fields (such as experience or education requirements) are not explicitly stated in the job description, infer them based on the context or leave them blank. Ensure that both the technical and soft skills are captured comprehensively.

    Text: {job_text}
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an experienced recruiter tasked with extracting complete and structured job information from unstructured text."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000,  # Increase token limit to handle longer responses
        n=1,
        stop=None,
        temperature=0.5,  # Reduce temperature for more deterministic output
    )

    json_output = response['choices'][0]['message']['content'].strip()

    # Ensure the output is valid JSON
    try:
        # Fix common issues with the JSON format
        json_output = json_output.replace("```json", "").replace("```", "").strip()
        json_output = json_output.replace("\n", "").replace("\\", "")
        json_output = json.loads(json_output)

    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return {}

    return json_output

import openai
import json
import parser 
from dateutil.relativedelta import relativedelta

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
    url = 'http://10.82.213.205:9013/v1/completions'
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

def gpt3_generate_summary(information):
    prompt = f"""
    Based on the following candidate information, generate a concise summary and a recommendation for a recruiter:
    Information: {json.dumps(information, indent=2)}

    Summary:
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an experienced recruiter, creating a summary and recommendation for a candidate based on their information."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.7,
    )

    summary = response['choices'][0]['message']['content'].strip()
    return summary
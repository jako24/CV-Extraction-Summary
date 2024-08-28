import csv
import re

# Function to remove commas inside quotes
def remove_commas_inside_quotes(text):
    # Match anything inside double quotes
    return re.sub(r'\"(.*?)\"', lambda m: m.group(0).replace(',', ''), text)

# Process the CSV file
input_csv_file = '/Users/janekkorczynski/CVExtraction/TestAccuracyFail.csv'
output_csv_file = '/Users/janekkorczynski/CVExtraction/TestAccuracyFailNoCommas.csv'

with open(input_csv_file, 'r', newline='', encoding='utf-8') as infile, open(output_csv_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for row in reader:
        # Clean each cell in the row
        cleaned_row = [remove_commas_inside_quotes(cell) for cell in row]
        # Write the cleaned row to the output CSV file
        writer.writerow(cleaned_row)

print(f"Processed CSV file has been saved to {output_csv_file}")

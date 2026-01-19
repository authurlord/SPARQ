import pandas as pd
import re

df = pd.read_csv('table.csv')

# Extract year from date strings using regex
def extract_year(date_str):
    match = re.search(r'\d{4}', date_str)
    return int(match.group()) if match else None

# Apply the function to each cell in the date columns
years = []
for col in df.columns:
    for date_str in df[col]:
        year = extract_year(str(date_str))
        if year is not None:
            years.append(year)

# Count how many years are 1990 or later
count_1990_or_later = sum(1 for year in years if year >= 1990)

print(f"Final Answer: {count_1990_or_later}")
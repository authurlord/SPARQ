import pandas as pd
import re

df = pd.read_csv('table.csv')

# Extract year from each date string in the first column (assuming all rows have date strings)
date_strings = df.iloc[:, 0].astype(str).str.extract(r'(\d{4})').dropna()

# Convert to integer and filter years >= 1990
years = pd.to_numeric(date_strings, errors='coerce')
count_1990_or_later = (years >= 1990).sum()

print(f"Final Answer: {count_1990_or_later}")
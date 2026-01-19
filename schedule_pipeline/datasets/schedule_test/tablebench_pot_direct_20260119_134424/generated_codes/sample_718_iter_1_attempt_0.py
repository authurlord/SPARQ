import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'research funding (000)' to numeric, handling any potential non-numeric entries
df['research funding (000)'] = pd.to_numeric(df['research funding (000)'], errors='coerce')

# Find the university with the highest research funding
max_funding_row = df.loc[df['research funding (000)'].idxmax()]
highest_university = max_funding_row['institution']
highest_funding = max_funding_row['research funding (000)']

# Find the university with the lowest research funding
min_funding_row = df.loc[df['research funding (000)'].idxmin()]
lowest_funding = min_funding_row['research funding (000)']

# Calculate the difference
difference = highest_funding - lowest_funding

print(f"Final Answer: {highest_university}, {difference}")
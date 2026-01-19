import pandas as pd

df = pd.read_csv('table.csv')
# Find the university with the highest research funding
max_funding_row = df.loc[df['research funding (000)'].idxmax()]
max_university = max_funding_row['institution']
max_funding = max_funding_row['research funding (000)']

# Find the university with the lowest research funding
min_funding_row = df.loc[df['research funding (000)'].idxmin()]
min_funding = min_funding_row['research funding (000)']

# Calculate the difference
difference = max_funding - min_funding

print(f"Final Answer: {max_university}, {difference}")
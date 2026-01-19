import pandas as pd

df = pd.read_csv('table.csv')
# Extract research funding in thousands
research_funding = df['research funding (000)']

# Find the university with the highest and lowest research funding
max_funding = research_funding.max()
min_funding = research_funding.min()
difference = max_funding - min_funding

# Get the names of universities with max and min funding
highest_university = df.loc[research_funding.idxmax(), 'institution']
lowest_university = df.loc[research_funding.idxmin(), 'institution']

print(f"Final Answer: {highest_university}, {lowest_university}, {difference}")
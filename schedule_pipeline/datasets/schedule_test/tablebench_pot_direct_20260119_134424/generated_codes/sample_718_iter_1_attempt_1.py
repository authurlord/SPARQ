import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'research funding (000)' to numeric, handling any potential non-numeric entries
df['research funding (000)'] = pd.to_numeric(df['research funding (000)'], errors='coerce')

# Find the university with the highest research funding
max_funding_uni = df.loc[df['research funding (000)'].idxmax(), 'institution']
min_funding_uni = df.loc[df['research funding (000)'].idxmin(), 'institution']

# Calculate the difference
difference = df['research funding (000)'].max() - df['research funding (000)'].min()

print(f"Final Answer: {max_funding_uni}, {difference}")
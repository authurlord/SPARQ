import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'research funding (000)' to numeric for calculations
df['research funding (000)'] = pd.to_numeric(df['research funding (000)'])
# Find the university with the highest research funding
max_funding_uni = df.loc[df['research funding (000)'].idxmax(), 'institution']
# Get the highest and lowest funding values
max_funding = df['research funding (000)'].max()
min_funding = df['research funding (000)'].min()
# Calculate the difference
difference = max_funding - min_funding
print(f"Final Answer: {max_funding_uni}, {difference}")
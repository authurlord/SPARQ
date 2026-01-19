import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'research funding (000)' to numeric, coercing errors to NaN if any
df['research funding (000)'] = pd.to_numeric(df['research funding (000)'], errors='coerce')

# Find the university with the highest and lowest research funding
max_funding = df.loc[df['research funding (000)'].idxmax(), 'institution']
min_funding = df.loc[df['research funding (000)'].idxmin(), 'institution']
max_value = df['research funding (000)'].max()
min_value = df['research funding (000)'].min()
difference = max_value - min_value

print(f"Final Answer: {max_funding}, {min_funding}, {difference}")
import pandas as pd

df = pd.read_csv('table.csv')
# Convert research funding to integer
df['research funding (000)'] = pd.to_numeric(df['research funding (000)'], errors='coerce')

# Find max and min research funding
max_funding = df['research funding (000)'].max()
min_funding = df['research funding (000)'].min()
difference = max_funding - min_funding

# Get the university with highest and lowest funding
highest_uni = df[df['research funding (000)'] == max_funding]['institution'].values[0]
lowest_uni = df[df['research funding (000)'] == min_funding]['institution'].values[0]

print(f"Final Answer: {highest_uni}, {difference}")
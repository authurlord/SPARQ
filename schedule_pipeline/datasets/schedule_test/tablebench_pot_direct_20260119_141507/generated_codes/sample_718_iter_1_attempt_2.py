import pandas as pd

df = pd.read_csv('table.csv')
# Convert research funding to integer
df['research funding (000)'] = pd.to_numeric(df['research funding (000)'], errors='coerce')

# Find max and min research funding values
max_funding = df['research funding (000)'].max()
min_funding = df['research funding (000)'].min()

# Get the institution with highest and lowest funding
highest_university = df[df['research funding (000)'] == max_funding]['institution'].values[0]
lowest_university = df[df['research funding (000)'] == min_funding]['institution'].values[0]

# Calculate the difference
difference = max_funding - min_funding

print(f"Final Answer: {highest_university}, {difference}")
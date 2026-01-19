import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where location is 'london'
london_universities = df[df['location'] == 'london']
# Sum the research funding (in thousands)
total_funding = london_universities['research funding (000)'].sum()
print(f"Final Answer: {total_funding}")
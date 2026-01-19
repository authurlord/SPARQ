import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where location is 'london'
london_unis = df[df['location'] == 'london']
# Sum the research funding for London universities
total_funding = london_unis['research funding (000)'].sum()
print(f"Final Answer: {total_funding}")
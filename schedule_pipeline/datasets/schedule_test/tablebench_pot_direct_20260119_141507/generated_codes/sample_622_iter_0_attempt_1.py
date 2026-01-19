import pandas as pd

df = pd.read_csv('table.csv')
# Filter universities located in London
london_universities = df[df['location'] == 'london']
# Sum the research funding for these universities
total_funding = london_universities['research funding (000)'].sum()
print(f"Final Answer: {total_funding}")
import pandas as pd

df = pd.read_csv('table.csv')
# Filter institutions located in London
london_institutions = df[df['location'] == 'london']
# Sum the research funding for these institutions
total_funding = london_institutions['research funding (000)'].sum()
print(f"Final Answer: {total_funding}")
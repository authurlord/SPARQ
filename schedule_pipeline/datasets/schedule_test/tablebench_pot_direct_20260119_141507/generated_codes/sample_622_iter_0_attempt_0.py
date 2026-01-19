import pandas as pd

df = pd.read_csv('table.csv')
# Filter universities located in London
london_universities = df[df['location'].str.contains('london', case=False)]
# Sum the research funding in thousands
total_funding = london_universities['research funding (000)'].sum()
print(f"Final Answer: {total_funding}")
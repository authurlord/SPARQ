import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where location is 'london'
london_universities = df[df['location'] == 'london']
# Count the number of universities in London
num_london_unis = len(london_universities)
print(f"Final Answer: {num_london_unis}")
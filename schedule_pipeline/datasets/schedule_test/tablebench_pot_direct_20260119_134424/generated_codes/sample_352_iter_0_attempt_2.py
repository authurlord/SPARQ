import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where location is 'london'
london_universities = df[df['location'] == 'london']
# Count the number of universities in London
count_london = len(london_universities)
print(f"Final Answer: {count_london}")
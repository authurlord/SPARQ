import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Class is '10BR'
br_filtered = df[df['Class'] == '10BR']
# Group by 'Year' and count the number of works
year_works_count = br_filtered.groupby('Year').size()
# Find the year with the maximum number of works
max_year = year_works_count.idxmax()
print(f"Final Answer: {max_year}")
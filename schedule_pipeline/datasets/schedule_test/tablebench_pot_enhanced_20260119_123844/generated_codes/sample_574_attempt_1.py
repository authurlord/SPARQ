import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Class is '10BR'
df_10br = df[df['Class'] == '10BR']
# Count number of works per year
works_count = df_10br.groupby('Year').size()
# Find the year with the maximum count
max_year = works_count.idxmax()
print(f"Final Answer: {max_year}")
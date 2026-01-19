import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Class is '10BR'
df_10br = df[df['Class'] == '10BR']
# Group by Year and count the number of works
works_by_year = df_10br.groupby('Year').size()
# Find the year with the maximum number of works
max_year = works_by_year.idxmax()
print(f"Final Answer: {max_year}")
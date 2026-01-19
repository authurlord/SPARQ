import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Class is '10BR'
df_10br = df[df['Class'] == '10BR']
# Count the number of works per year
works_by_year = df_10br['Year'].value_counts()
# Find the year with the most works
max_year = works_by_year.idxmax()
print(f"Final Answer: {max_year}")
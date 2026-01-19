import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Class is '10BR'
ten_br_df = df[df['Class'] == '10BR']
# Count number of works per year
works_count = ten_br_df['Year'].value_counts()
# Find the year with the most works
most_works_year = works_count.idxmax()
print(f"Final Answer: {most_works_year}")
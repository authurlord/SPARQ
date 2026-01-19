import pandas as pd

df = pd.read_csv('table.csv')
# Filter for '10BR' class
ten_br_df = df[df['Class'] == '10BR']
# Count works per year
works_count = ten_br_df['Year'].value_counts()
# Find the year with the most works
max_year = works_count.idxmax()
print(f"Final Answer: {max_year}")
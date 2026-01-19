import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Class is '10BR'
br_class = df[df['Class'] == '10BR']
# Group by Year and count the number of works
year_works = br_class.groupby('Year').size()
# Find the year with the maximum count
max_year = year_works.idxmax()
print(f"Final Answer: {max_year}")
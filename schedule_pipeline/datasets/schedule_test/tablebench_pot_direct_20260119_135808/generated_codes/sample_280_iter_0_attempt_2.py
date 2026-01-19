import pandas as pd

df = pd.read_csv('table.csv')
# Filter for healthcare and education industries
filtered_df = df[(df['industry'] == 'healthcare') | (df['industry'] == 'education')]
# Sum the 2012 employees
total_employees_2012 = filtered_df['2012 employees (total)'].astype(int).sum()
print(f"Final Answer: {total_employees_2012}")
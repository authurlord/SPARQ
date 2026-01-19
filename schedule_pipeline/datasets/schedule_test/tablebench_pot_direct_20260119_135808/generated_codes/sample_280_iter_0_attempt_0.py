import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where industry is 'healthcare' or 'education'
filtered_df = df[(df['industry'] == 'healthcare') | (df['industry'] == 'education')]
# Sum the '2012 employees (total)' column
total_employees_2012 = filtered_df['2012 employees (total)'].sum()
print(f"Final Answer: {total_employees_2012}")
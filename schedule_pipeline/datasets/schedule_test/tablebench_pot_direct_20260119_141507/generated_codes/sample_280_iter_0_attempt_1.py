import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where industry is 'healthcare' or 'education'
healthcare_education = df[(df['industry'] == 'healthcare') | (df['industry'] == 'education')]
# Sum the 2012 employees for these filtered rows
total_employees = healthcare_education['2012 employees (total)'].sum()
print(f"Final Answer: {total_employees}")
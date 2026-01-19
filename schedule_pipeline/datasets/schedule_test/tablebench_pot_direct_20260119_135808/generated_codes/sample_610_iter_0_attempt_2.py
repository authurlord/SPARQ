import pandas as pd

df = pd.read_csv('table.csv')
# Filter for males (row where SPECIFICATION_1 is 'A.')
males_row = df[(df['SPECIFICATION'] == 'I.') & (df['SPECIFICATION_1'] == 'A.')]
# Extract population for 20-29 and 30-39 age groups
males_20_29 = males_row['POPULATION (by age group in 2002)_3'].values[0]
males_30_39 = males_row['POPULATION (by age group in 2002)_4'].values[0]
# Calculate total
total_males = males_20_29 + males_30_39
print(f"Final Answer: {total_males}")
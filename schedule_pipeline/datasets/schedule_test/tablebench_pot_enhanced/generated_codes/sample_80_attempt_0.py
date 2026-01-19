import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for year 2001
row_2001 = df[df['Year'] == '2001']
# Extract the value for 'Naturalisation by marriage'
naturalisation_by_marriage_2001 = row_2001['Naturalisation by marriage'].values[0]
print(f"Final Answer: {naturalisation_by_marriage_2001}")
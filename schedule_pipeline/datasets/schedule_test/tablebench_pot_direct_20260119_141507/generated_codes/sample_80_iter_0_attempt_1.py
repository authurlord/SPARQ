import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for year 2001 and get the value for 'Naturalisation by marriage'
marriage_naturalisation_2001 = df[df['Year'] == '2001']['Naturalisation by marriage'].values[0]
print(f"Final Answer: {marriage_naturalisation_2001}")
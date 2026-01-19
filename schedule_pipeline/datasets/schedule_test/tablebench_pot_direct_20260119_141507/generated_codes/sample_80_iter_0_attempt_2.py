import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for year 2001 and get the value in "Naturalisation by marriage" column
value_2001_marriage = df[df['Year'] == '2001']['Naturalisation by marriage'].values[0]
print(f"Final Answer: {value_2001_marriage}")
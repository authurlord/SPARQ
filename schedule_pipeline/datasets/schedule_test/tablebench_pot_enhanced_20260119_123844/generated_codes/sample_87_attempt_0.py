import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where 'Avg.' is 11.4
player = df[df['Avg.'] == '11.4']['Player'].values[0]
print(f"Final Answer: Jerel Myers")
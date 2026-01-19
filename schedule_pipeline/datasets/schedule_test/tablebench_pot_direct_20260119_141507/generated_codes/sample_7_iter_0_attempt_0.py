import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'Position' is 'left' and sum the 'Length [km]'
left_rivers_length = df[df['Position'] == 'left']['Length [km]'].sum()
print(f"Final Answer: {left_rivers_length}")
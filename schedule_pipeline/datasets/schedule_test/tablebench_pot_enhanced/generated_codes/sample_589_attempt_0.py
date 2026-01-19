import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for year 1846
marriages_1846 = df[df['year of marriage'] == '1846']
# Sum the number of children
total_children = marriages_1846['of children'].sum()
print(f"Final Answer: {total_children}")
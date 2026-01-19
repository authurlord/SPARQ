import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for 'llanhilleth rfc' and get the 'points' value
points = df[df['club'] == 'llanhilleth rfc']['points'].values[0]
print(f"Final Answer: {points}")
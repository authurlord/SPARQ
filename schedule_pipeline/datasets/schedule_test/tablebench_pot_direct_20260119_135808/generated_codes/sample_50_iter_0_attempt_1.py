import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for Wattstown RFC
wattstown_points = df[df['club'] == 'wattstown rfc']['points'].iloc[0]
print(f"Final Answer: {wattstown_points}")
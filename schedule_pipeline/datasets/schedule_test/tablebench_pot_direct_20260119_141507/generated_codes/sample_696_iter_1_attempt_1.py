import pandas as pd

df = pd.read_csv('table.csv')

# Extract the 'foreign' language row (it's a row in the table, not a column)
foreign_values = df.loc[df['language'] == 'foreign'].iloc[0].dropna()

# Find the municipality with the highest and lowest values
max_value = foreign_values.max()
min_value = foreign_values.min()
max_municipality = foreign_values.idxmax()
min_municipality = foreign_values.idxmin()

difference = max_value - min_value

print(f"Final Answer: {max_municipality}, {difference:.1f}")
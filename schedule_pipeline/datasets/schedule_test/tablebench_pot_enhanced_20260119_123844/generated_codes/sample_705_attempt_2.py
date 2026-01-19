import pandas as pd

df = pd.read_csv('table.csv')
# Extract the row for 'Copper (mg)'
copper_row = df[df['STAPLE:'] == 'Copper (mg)']
# Drop the 'STAPLE:' column and find the column with the maximum value
copper_values = copper_row.iloc[0, 1:]  # Exclude the first column (STAPLE:)
max_copper_staple = copper_values.idxmax()
print(f"Final Answer: {max_copper_staple}")
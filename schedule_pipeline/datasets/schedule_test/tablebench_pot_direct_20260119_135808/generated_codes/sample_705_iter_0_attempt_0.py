import pandas as pd

df = pd.read_csv('table.csv')
# Extract the row for 'Copper (mg)'
copper_row = df.loc[17]
# Get the staple foods (columns except the first one)
staples = df.columns[1:]
copper_values = copper_row[1:].astype(float)
# Find the staple with the highest copper value
max_copper_staple = staples[copper_values.idxmax()]
print(f"Final Answer: {max_copper_staple}")
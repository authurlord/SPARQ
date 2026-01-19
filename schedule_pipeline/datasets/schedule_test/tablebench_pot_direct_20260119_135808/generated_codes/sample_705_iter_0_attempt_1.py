import pandas as pd

df = pd.read_csv('table.csv')
# Extract the row for 'Copper (mg)'
copper_row = df[df['STAPLE:'] == 'Copper (mg)']
# Get the values for each staple food
copper_values = copper_row.iloc[0, 1:]  # Exclude the first column (STAPLE:)
staples = df.columns[1:]  # Get staple food names
# Find the staple with the highest copper content
max_copper_staple = staples[copper_values.idxmax()]
print(f"Final Answer: {max_copper_staple}")
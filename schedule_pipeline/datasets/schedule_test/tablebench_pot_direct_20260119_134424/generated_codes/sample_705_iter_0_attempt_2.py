import pandas as pd

df = pd.read_csv('table.csv')
# Extract the row where the component is 'Copper (mg)'
copper_row = df[df['STAPLE:'] == 'Copper (mg)']
# Get the values for each staple food (excluding the first column which is 'STAPLE:')
staples = df.columns[1:]
copper_values = copper_row.iloc[0, 1:].astype(float)
# Find the staple with the highest copper content
max_copper_staple = staples[copper_values.idxmax()]
print(f"Final Answer: {max_copper_staple}")
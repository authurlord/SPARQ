import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract the row for 'Copper (mg)'
copper_row = df[df['STAPLE:'] == 'Copper (mg)']

# Get the columns (staple foods) and their values
staples = df.columns[1:]  # Skip the first column 'STAPLE:'
copper_values = copper_row.iloc[0, 1:].astype(float, errors='ignore')  # Convert to float, ignore invalid

# Replace non-numeric values (like '-') with NaN
copper_values = pd.to_numeric(copper_values, errors='coerce')

# Find the staple with the highest copper value
max_copper_staple = staples[copper_values.idxmax()]

print(f"Final Answer: {max_copper_staple}")
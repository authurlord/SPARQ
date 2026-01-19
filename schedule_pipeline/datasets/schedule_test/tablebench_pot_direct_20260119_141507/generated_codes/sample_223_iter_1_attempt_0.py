import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'mccain %' from string with '%' to float
df['mccain %'] = df['mccain %'].str.replace('%', '').astype(float)

# Ensure 'total' is numeric
df['total'] = pd.to_numeric(df['total'], errors='coerce')

# Calculate the correlation between mccain % and total
correlation = df['mccain %'].corr(df['total'])

print(f"Final Answer: {correlation:.3f}")
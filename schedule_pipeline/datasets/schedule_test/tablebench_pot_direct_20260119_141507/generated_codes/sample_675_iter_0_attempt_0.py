import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Clean the 'capacity in use' column: remove % and convert to float
df['capacity in use'] = df['capacity in use'].str.replace('%', '').astype(float)

# Compute correlation between 'total passengers' and 'capacity in use'
correlation = df['total passengers'].corr(df['capacity in use'])

print(f"Final Answer: {correlation:.2f}")
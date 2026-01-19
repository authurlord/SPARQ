import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Clean the 'annual change' and 'capacity in use' columns by removing '%' and converting to float
df['annual change'] = df['annual change'].str.rstrip('%').astype(float)
df['capacity in use'] = df['capacity in use'].str.replace(',', '').str.rstrip('%').astype(float)

# Calculate the correlation between annual change and capacity in use
correlation = df['annual change'].corr(df['capacity in use'])

print(f"Final Answer: {correlation:.2f}")
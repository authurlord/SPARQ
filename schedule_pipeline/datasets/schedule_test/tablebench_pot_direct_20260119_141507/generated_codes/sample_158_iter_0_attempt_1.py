import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean the 'capacity in use' column: remove spaces and commas, convert to float
df['capacity in use'] = df['capacity in use'].str.replace(',', '').str.replace('%', '').astype(float)

# Convert 'annual change' to float (already in percentage format)
df['annual change'] = df['annual change'].str.replace('%', '').astype(float)

# Calculate the correlation between 'annual change' and 'capacity in use'
correlation = df['annual change'].corr(df['capacity in use'])

print(f"Final Answer: {correlation:.2f}")
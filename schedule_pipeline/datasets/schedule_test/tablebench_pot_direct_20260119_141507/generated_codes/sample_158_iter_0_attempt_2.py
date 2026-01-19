import pandas as pd
import numpy as np

# Load the dataframe
df = pd.read_csv('table.csv')

# Clean the 'annual change' and 'capacity in use' columns by removing commas and converting to float
df['annual change'] = df['annual change'].str.replace('%', '').astype(float)
df['capacity in use'] = df['capacity in use'].str.replace(',', '').str.replace('%', '').astype(float)

# Calculate the correlation between 'annual change' and 'capacity in use'
correlation = df['annual change'].corr(df['capacity in use'])

print(f"Final Answer: {correlation:.2f}")
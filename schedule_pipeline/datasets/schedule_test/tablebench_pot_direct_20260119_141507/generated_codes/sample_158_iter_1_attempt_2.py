import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean 'capacity in use' column: remove space and comma, and convert to float
df['capacity in use'] = df['capacity in use'].str.replace(' ', '').str.replace(',', '.').str.rstrip('%').astype(float)

# Convert 'annual change' to float (remove %)
df['annual change'] = df['annual change'].str.rstrip('%').astype(float)

# Compute correlation between annual change and capacity in use
correlation = df['annual change'].corr(df['capacity in use'])

print(f"Final Answer: {correlation:.2f}")
import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean 'annual change' column: extract numeric value from string like '9.24%'
df['annual change'] = df['annual change'].str.replace('%', '').astype(float)

# Clean 'capacity in use' column: remove space and comma, then convert to float
df['capacity in use'] = df['capacity in use'].str.replace(' , ', '.').str.replace(' %', '').str.replace(',', '.').astype(float)

# Compute the correlation between annual change and capacity in use
correlation = df['annual change'].corr(df['capacity in use'])

print(f"Final Answer: {correlation:.2f}")
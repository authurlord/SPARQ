import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean the 'capacity in use' column: remove % and replace comma with dot
df['capacity in use'] = df['capacity in use'].str.replace(',', '.').str.replace('%', '').astype(float)

# Compute the correlation between 'total passengers' and 'capacity in use'
correlation = df['total passengers'].corr(df['capacity in use'])

print(f"Final Answer: {correlation:.2f}")
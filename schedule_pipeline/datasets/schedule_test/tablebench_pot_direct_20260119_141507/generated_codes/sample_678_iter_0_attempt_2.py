import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert '% change' column to numeric, handling any potential formatting issues
df['% change'] = df['% change'].str.replace('%', '').astype(float)

# Calculate the correlation between 'density (pop / km square)' and '% change'
correlation = df['density (pop / km square)'].corr(df['% change'])

print(f"Final Answer: {correlation:.3f}")
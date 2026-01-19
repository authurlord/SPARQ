import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'bp comp 1 (˚C)' and '% wt comp 1' to numeric, handling errors by converting strings like '- 0.5'
df['bp comp 1 (˚c)'] = pd.to_numeric(df['bp comp 1 (˚c)'], errors='coerce')
df['% wt comp 1'] = pd.to_numeric(df['% wt comp 1'], errors='coerce')

# Drop rows with NaN values after conversion
df = df.dropna(subset=['bp comp 1 (˚c)', '% wt comp 1'])

# Calculate correlation coefficient
correlation = df['bp comp 1 (˚c)'].corr(df['% wt comp 1'])

print(f"Final Answer: {correlation:.3f}")
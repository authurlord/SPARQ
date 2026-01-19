import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'Death toll' column to numeric by removing commas and converting to float
df['Death toll'] = df['Death toll'].str.replace(',', '').astype(float)

# Calculate the correlation between 'Magnitude' and 'Death toll'
correlation = df['Magnitude'].corr(df['Death toll'])

print(f"Final Answer: {correlation:.3f}")
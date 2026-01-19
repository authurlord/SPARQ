import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'death toll' from string to numeric (remove commas)
df['Death toll'] = df['Death toll'].str.replace(',', '').astype(float)

# Calculate correlation between 'Magnitude' and 'Death toll'
correlation = df['Magnitude'].corr(df['Death toll'])

print(f"Final Answer: {correlation:.3f}")
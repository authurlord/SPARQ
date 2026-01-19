import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'death toll' from string to numeric, removing commas
df['Death toll'] = df['Death toll'].str.replace(',', '').astype(float)

# Extract the 'Magnitude' and 'Death toll' columns
magnitude = df['Magnitude'].astype(float)
death_toll = df['Death toll']

# Calculate the correlation coefficient
correlation_coefficient = np.corrcoef(magnitude, death_toll)[0, 1]
print(f"Final Answer: {correlation_coefficient:.3f}")
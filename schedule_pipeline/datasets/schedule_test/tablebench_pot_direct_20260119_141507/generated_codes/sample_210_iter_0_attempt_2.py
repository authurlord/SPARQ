import pandas as pd

df = pd.read_csv('table.csv')
# Extract elevation and prominence columns
elevation = df['elevation (m)'].astype(float)
prominence = df['prominence (m)'].astype(float)

# Calculate the correlation between elevation and prominence
correlation = elevation.corr(prominence)
print(f"Final Answer: {correlation:.3f}")
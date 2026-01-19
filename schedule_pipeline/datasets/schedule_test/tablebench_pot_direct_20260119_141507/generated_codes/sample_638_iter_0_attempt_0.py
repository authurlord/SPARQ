import pandas as pd

df = pd.read_csv('table.csv')
# Extract the relevant columns
elevation = df['elevation (m)']
prominence = df['prominence (m)']

# Calculate the correlation coefficient
correlation_coefficient = elevation.corr(prominence)

print(f"Final Answer: {correlation_coefficient:.3f}")
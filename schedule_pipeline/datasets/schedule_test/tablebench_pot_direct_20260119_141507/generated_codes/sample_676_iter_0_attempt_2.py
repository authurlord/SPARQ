import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between 'eccentricity' and 'period (h)'
correlation = df['eccentricity'].corr(df['period (h)'])
print(f"Final Answer: {correlation:.3f}")
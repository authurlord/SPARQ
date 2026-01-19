import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'])
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'])

# Calculate Pearson correlation coefficient
correlation = df['elevation (m)'].corr(df['prominence (m)'])

# Output the result
print(f"Final Answer: {correlation:.3f}")
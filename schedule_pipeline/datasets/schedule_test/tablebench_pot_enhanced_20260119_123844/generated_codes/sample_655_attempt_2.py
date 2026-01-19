import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric for correlation calculation
df['p1 diameter (mm)'] = pd.to_numeric(df['p1 diameter (mm)'])
df['p max (bar)'] = pd.to_numeric(df['p max (bar)'])

# Calculate correlation coefficient
correlation = df['p1 diameter (mm)'].corr(df['p max (bar)'])
print(f"Final Answer: {correlation:.4f}")
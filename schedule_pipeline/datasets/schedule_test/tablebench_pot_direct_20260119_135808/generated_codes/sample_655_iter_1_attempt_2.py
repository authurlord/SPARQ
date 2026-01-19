import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'p1 diameter (mm)' and 'p max (bar)' to numeric, coercing errors to NaN
df['p1 diameter (mm)'] = pd.to_numeric(df['p1 diameter (mm)'], errors='coerce')
df['p max (bar)'] = pd.to_numeric(df['p max (bar)'], errors='coerce')

# Drop rows with NaN values in either column
df.dropna(subset=['p1 diameter (mm)', 'p max (bar)'], inplace=True)

# Calculate the correlation coefficient
correlation = df['p1 diameter (mm)'].corr(df['p max (bar)'])

print(f"Final Answer: {correlation:.4f}")
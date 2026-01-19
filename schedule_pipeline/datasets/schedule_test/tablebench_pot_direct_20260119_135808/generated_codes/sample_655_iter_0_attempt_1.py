import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric, coercing errors to NaN
df['p1 diameter (mm)'] = pd.to_numeric(df['p1 diameter (mm)'], errors='coerce')
df['p max (bar)'] = pd.to_numeric(df['p max (bar)'], errors='coerce')

# Drop rows with NaN values in the selected columns
df_clean = df.dropna(subset=['p1 diameter (mm)', 'p max (bar)'])

# Calculate correlation coefficient
correlation = df_clean['p1 diameter (mm)'].corr(df_clean['p max (bar)'])

print(f"Final Answer: {correlation:.4f}")
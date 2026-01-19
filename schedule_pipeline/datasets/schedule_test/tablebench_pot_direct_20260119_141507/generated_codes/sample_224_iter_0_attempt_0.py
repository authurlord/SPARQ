import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'p1 diameter (mm)' to numeric and remove non-numeric parts
df['p1 diameter (mm)'] = pd.to_numeric(df['p1 diameter (mm)'].str.replace('x', ''), errors='coerce')

# Ensure 'p max (bar)' is numeric
df['p max (bar)'] = pd.to_numeric(df['p max (bar)'], errors='coerce')

# Drop rows with missing values
df.dropna(subset=['p1 diameter (mm)', 'p max (bar)'], inplace=True)

# Compute correlation between p1 diameter and p max
correlation = df['p max (bar)'].corr(df['p1 diameter (mm)'])

print(f"Final Answer: {correlation:.3f}")
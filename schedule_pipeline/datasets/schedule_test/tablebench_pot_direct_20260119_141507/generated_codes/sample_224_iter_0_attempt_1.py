import pandas as pd

df = pd.read_csv('table.csv')

# Convert p1 diameter (mm) to numeric and ensure p max is numeric
df['p1 diameter (mm)'] = pd.to_numeric(df['p1 diameter (mm)'], errors='coerce')
df['p max ( bar )'] = pd.to_numeric(df['p max ( bar )'], errors='coerce')

# Drop rows with missing values
df_clean = df.dropna(subset=['p1 diameter (mm)', 'p max ( bar )'])

# Calculate the correlation between p1 diameter and p max
correlation = df_clean['p1 diameter (mm)'].corr(df_clean['p max ( bar )'])

print(f"Final Answer: {correlation:.3f}")
import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'sales (billion )' and 'profits (billion )' columns to numeric
df['sales (billion )'] = pd.to_numeric(df['sales (billion )'], errors='coerce')
df['profits (billion )'] = pd.to_numeric(df['profits (billion )'], errors='coerce')

# Calculate the correlation coefficient
correlation = df['sales (billion )'].corr(df['profits (billion )'])

print(f"Final Answer: {correlation:.4f}")
import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'pop (2010)' and 'land ( sqmi )' to numeric, handling any potential errors
df['pop (2010)'] = pd.to_numeric(df['pop (2010)'], errors='coerce')
df['land ( sqmi )'] = pd.to_numeric(df['land ( sqmi )'], errors='coerce')

# Drop rows with NaN after conversion
df = df.dropna(subset=['pop (2010)', 'land ( sqmi )'])

# Compute the correlation coefficient
correlation = df['pop (2010)'].corr(df['land ( sqmi )'])
print(f"Final Answer: {correlation:.3f}")
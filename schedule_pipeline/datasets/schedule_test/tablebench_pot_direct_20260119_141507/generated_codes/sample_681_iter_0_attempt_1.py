import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'pop (2010)' to numeric, handling any non-numeric values
df['pop (2010)'] = pd.to_numeric(df['pop (2010)'], errors='coerce')

# Drop rows with NaN due to conversion issues
df = df.dropna(subset=['pop (2010)', 'land ( sqmi )'])

# Calculate the correlation coefficient between 'pop (2010)' and 'land ( sqmi )'
correlation = df['pop (2010)'].corr(df['land ( sqmi )'])
print(f"Final Answer: {correlation:.3f}")
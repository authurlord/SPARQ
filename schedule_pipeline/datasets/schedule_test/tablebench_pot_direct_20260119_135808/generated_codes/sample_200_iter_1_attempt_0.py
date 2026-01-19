import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'metres' and 'prominence (m)' to numeric
df['metres'] = pd.to_numeric(df['metres'], errors='coerce')
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'], errors='coerce')

# Drop rows with missing values
df.dropna(inplace=True)

# Calculate correlation coefficient
correlation = df['metres'].corr(df['prominence (m)'])

print(f"Final Answer: {correlation:.3f}")
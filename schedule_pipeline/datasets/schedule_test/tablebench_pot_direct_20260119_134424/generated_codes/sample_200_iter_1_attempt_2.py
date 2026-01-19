import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'metres' and 'prominence (m)' to numeric
df['metres'] = pd.to_numeric(df['metres'])
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'])

# Calculate correlation
correlation = df['metres'].corr(df['prominence (m)'])

print(f"Final Answer: {correlation:.2f}")
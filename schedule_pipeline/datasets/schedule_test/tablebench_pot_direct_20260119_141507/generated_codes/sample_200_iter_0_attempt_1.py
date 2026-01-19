import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'metres' and 'prominence (m)' to numeric
df['metres'] = pd.to_numeric(df['metres'], errors='coerce')
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'], errors='coerce')

# Drop rows with NaN values due to conversion errors
df = df.dropna(subset=['metres', 'prominence (m)'])

# Compute the correlation between height and prominence
correlation = df['metres'].corr(df['prominence (m)'])
print(f"Final Answer: {correlation:.3f}")
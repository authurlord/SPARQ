import pandas as pd

df = pd.read_csv('table.csv')

# Convert elevation and prominence to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'], errors='coerce')
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'], errors='coerce')

# Remove rows with missing values
df = df.dropna(subset=['elevation (m)', 'prominence (m)'])

# Calculate correlation between elevation and prominence
correlation = df['elevation (m)'].corr(df['prominence (m)'])

# Print the correlation value and a brief interpretation
print(f"Final Answer: {correlation:.3f}")
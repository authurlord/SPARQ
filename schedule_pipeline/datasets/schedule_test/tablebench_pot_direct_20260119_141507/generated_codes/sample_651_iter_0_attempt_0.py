import pandas as pd

df = pd.read_csv('table.csv')
# Convert elevation and prominence to numeric, handling any parsing errors
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'], errors='coerce')
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'], errors='coerce')

# Drop rows with NaN due to conversion errors
df_clean = df.dropna(subset=['elevation (m)', 'prominence (m)'])

# Calculate correlation coefficient between elevation and prominence
correlation_coefficient = df_clean['elevation (m)'].corr(df_clean['prominence (m)'])
print(f"Final Answer: {correlation_coefficient:.3f}")
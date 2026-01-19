import pandas as pd

df = pd.read_csv('table.csv')

# Clean and convert columns to numeric
df['snow (days / year)'] = pd.to_numeric(df['snow (days / year)'].str.replace(' ', ''), errors='coerce')
df['storms (days / year)'] = pd.to_numeric(df['storms (days / year)'].str.replace(' ', ''), errors='coerce')
df['fog (days / year)'] = pd.to_numeric(df['fog (days / year)'].str.replace(' ', ''), errors='coerce')

# Drop rows with NaN values
df.dropna(inplace=True)

# Compute correlation coefficients
corr_storms = df['snow (days / year)'].corr(df['storms (days / year)'])
corr_fog = df['snow (days / year)'].corr(df['fog (days / year)'])

# Compare absolute correlation values
if abs(corr_storms) > abs(corr_fog):
    result = "storms"
else:
    result = "fog"

print(f"Final Answer: {result}")
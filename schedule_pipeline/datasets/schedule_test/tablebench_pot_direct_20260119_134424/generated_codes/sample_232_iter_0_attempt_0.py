import pandas as pd

df = pd.read_csv('table.csv')

# Clean and convert columns to numeric
df['snow (days / year)'] = df['snow (days / year)'].str.replace(' ', '').astype(int)
df['storms (days / year)'] = df['storms (days / year)'].str.replace(' ', '').astype(int)
df['fog (days / year)'] = df['fog (days / year)'].str.replace(' ', '').astype(int)

# Calculate correlation between snow days and storm days
corr_snow_storms = df['snow (days / year)'].corr(df['storms (days / year)'])

# Calculate correlation between snow days and fog days
corr_snow_fog = df['snow (days / year)'].corr(df['fog (days / year)'])

# Compare absolute correlation values
if abs(corr_snow_storms) > abs(corr_snow_fog):
    result = "storms"
else:
    result = "fog"

print(f"Final Answer: {result}")
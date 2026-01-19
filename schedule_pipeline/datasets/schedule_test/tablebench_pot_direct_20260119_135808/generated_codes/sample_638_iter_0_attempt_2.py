import pandas as pd

df = pd.read_csv('table.csv')
# Convert elevation and prominence to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'])
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'])

# Calculate correlation coefficient
correlation = df['elevation (m)'].corr(df['prominence (m)'])

# Determine significance based on magnitude
if abs(correlation) > 0.7:
    conclusion = "There is a significant correlation."
else:
    conclusion = "There is no significant correlation."

print(f"Final Answer: {conclusion}, correlation coefficient: {correlation:.3f}")
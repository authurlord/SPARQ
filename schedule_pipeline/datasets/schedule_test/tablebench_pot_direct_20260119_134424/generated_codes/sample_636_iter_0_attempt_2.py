import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['area (km square)'] = pd.to_numeric(df['area (km square)'])
df['gdp (billion us)'] = pd.to_numeric(df['gdp (billion us)'])

# Calculate correlation coefficient
correlation = df['area (km square)'].corr(df['gdp (billion us)'])

# Determine significance based on correlation coefficient
if abs(correlation) > 0.7:
    conclusion = "There is a significant correlation."
else:
    conclusion = "There is no significant correlation."

print(f"Final Answer: {correlation:.3f}")
import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['area (km square)'] = pd.to_numeric(df['area (km square)'])
df['gdp (billion us)'] = pd.to_numeric(df['gdp (billion us)'])

# Calculate the correlation coefficient
correlation = df['area (km square)'].corr(df['gdp (billion us)'])

# Determine significance based on the coefficient value
if abs(correlation) >= 0.7:
    conclusion = "There is a strong correlation."
elif abs(correlation) >= 0.3:
    conclusion = "There is a moderate correlation."
else:
    conclusion = "There is a weak correlation."

print(f"Final Answer: {conclusion}, correlation coefficient: {correlation:.3f}")
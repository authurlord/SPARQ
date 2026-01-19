import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'])
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'])

# Calculate correlation coefficient
correlation = df['elevation (m)'].corr(df['prominence (m)'])

# Interpretation: If |correlation| > 0.7, it's considered significant
if abs(correlation) > 0.7:
    conclusion = "Yes, there is a significant correlation."
else:
    conclusion = "No, there is not a significant correlation."

print(f"Final Answer: {conclusion}, {correlation:.3f}")
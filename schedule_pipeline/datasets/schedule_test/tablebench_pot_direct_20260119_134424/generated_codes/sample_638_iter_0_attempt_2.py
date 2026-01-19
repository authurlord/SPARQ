import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'])
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'])

# Calculate correlation coefficient
correlation = df['elevation (m)'].corr(df['prominence (m)'])

# Conclusion based on correlation coefficient
if abs(correlation) > 0.7:
    conclusion = "There is a significant correlation."
else:
    conclusion = "There is no significant correlation."

print(f"Correlation coefficient: {correlation:.3f}")
print(f"Final Answer: {conclusion}, {correlation:.3f}")
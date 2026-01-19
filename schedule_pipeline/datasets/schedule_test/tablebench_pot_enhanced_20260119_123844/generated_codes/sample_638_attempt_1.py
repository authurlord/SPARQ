import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'])
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'])

# Calculate correlation coefficient
correlation = df['elevation (m)'].corr(df['prominence (m)'])

# Determine significance based on magnitude
if abs(correlation) > 0.7:
    conclusion = "There is a strong correlation."
elif abs(correlation) > 0.3:
    conclusion = "There is a moderate correlation."
else:
    conclusion = "There is a weak correlation."

print(f"Correlation coefficient: {correlation:.3f}")
print(f"Conclusion: {conclusion}")
print(f"Final Answer: {correlation:.3f}")
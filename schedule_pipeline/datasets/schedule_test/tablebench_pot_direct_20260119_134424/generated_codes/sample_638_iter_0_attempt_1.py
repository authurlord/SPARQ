import pandas as pd

df = pd.read_csv('table.csv')
# Convert elevation and prominence to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'])
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'])

# Calculate correlation coefficient
correlation = df['elevation (m)'].corr(df['prominence (m)'])

# Conclusion based on correlation coefficient
if abs(correlation) >= 0.7:
    conclusion = "There is a strong correlation."
elif abs(correlation) >= 0.3:
    conclusion = "There is a moderate correlation."
else:
    conclusion = "There is a weak correlation."

print(f"Final Answer: {conclusion} Correlation coefficient: {correlation:.2f}")
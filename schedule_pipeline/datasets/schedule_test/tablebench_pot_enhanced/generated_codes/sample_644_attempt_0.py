import pandas as pd

df = pd.read_csv('table.csv')
# Convert boiling point and critical temperature to float
boiling_point = df[df['physical property'] == 'boiling point (degree)'].iloc[0, 1:].astype(float)
critical_temp = df[df['physical property'] == 'critical temperature (k)'].iloc[0, 1:].astype(float)

# Calculate correlation coefficient
correlation = boiling_point.corr(critical_temp)
print(f"Final Answer: {correlation:.3f}")
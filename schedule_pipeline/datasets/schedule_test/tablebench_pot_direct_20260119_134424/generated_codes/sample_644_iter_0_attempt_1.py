import pandas as pd

df = pd.read_csv('table.csv')
# Convert boiling point and critical temperature to float
boiling_point = df['boiling point (degree)'].astype(float)
critical_temp = df['critical temperature (k)'].astype(float)

# Calculate correlation coefficient
correlation = boiling_point.corr(critical_temp)
print(f"Final Answer: {correlation:.3f}")
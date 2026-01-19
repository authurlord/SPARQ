import pandas as pd

df = pd.read_csv('table.csv')

# Extract the relevant columns by their full names
boiling_point = df['boiling point (degree)'].astype(float)
critical_temp = df['critical temperature (k)'].astype(float)

# Compute the correlation coefficient
correlation = boiling_point.corr(critical_temp)

print(f"Final Answer: {correlation:.3f}")
import pandas as pd

df = pd.read_csv('table.csv')
# Convert the relevant columns to numeric
boiling_point = pd.to_numeric(df['boiling point (degree)'], errors='coerce')
critical_temp = pd.to_numeric(df['critical temperature (k)'], errors='coerce')

# Calculate correlation coefficient
correlation = boiling_point.corr(critical_temp)
print(f"Final Answer: {correlation:.4f}")
import pandas as pd

df = pd.read_csv('table.csv')
# Extract the relevant columns: boiling point (degree) and critical temperature (k)
boiling_point = df['boiling point (degree)'].astype(float)
critical_temp = df['critical temperature (k)'].astype(float)

# Calculate the correlation coefficient
correlation_coefficient = boiling_point.corr(critical_temp)
print(f"Final Answer: {correlation_coefficient:.3f}")
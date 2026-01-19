import pandas as pd

# Load the table
df = pd.read_csv('table.csv')

# Extract the relevant columns
boiling_point = df['boiling point (degree)'].astype(float)
critical_temperature = df['critical temperature (k)'].astype(float)

# Calculate the correlation coefficient
correlation_coefficient = boiling_point.corr(critical_temperature)

print(f"Final Answer: {correlation_coefficient:.3f}")
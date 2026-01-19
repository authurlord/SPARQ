import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'weight (kg / m)' to numeric, coercing errors to NaN if any
df['weight (kg / m)'] = pd.to_numeric(df['weight (kg / m)'], errors='coerce')
# Calculate the mean of the weight column, ignoring any invalid entries
average_weight = df['weight (kg / m)'].mean()
print(f"Final Answer: {average_weight:.2f}")
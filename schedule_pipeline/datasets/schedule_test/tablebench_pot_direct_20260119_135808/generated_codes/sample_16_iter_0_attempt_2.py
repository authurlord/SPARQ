import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'US Chart position' to numeric, coercing errors to NaN
df['US Chart position'] = pd.to_numeric(df['US Chart position'], errors='coerce')
# Calculate the mean of valid numeric values
average_position = df['US Chart position'].mean()
print(f"Final Answer: {average_position:.1f}")
import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'specific impulse (s)' to numeric, coercing errors to NaN
df['specific impulse (s)'] = pd.to_numeric(df['specific impulse (s)'], errors='coerce')
# Count engines with specific impulse > 1000
count_high_impulse = df[df['specific impulse (s)'] > 1000].shape[0]
print(f"Final Answer: {count_high_impulse}")
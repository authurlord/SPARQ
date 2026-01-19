import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'specific impulse (s)' to numeric to allow comparison
df['specific impulse (s)'] = pd.to_numeric(df['specific impulse (s)'], errors='coerce')
# Count engines with specific impulse > 1000 seconds
count = (df['specific impulse (s)'] > 1000).sum()
print(f"Final Answer: {count}")
import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'specific impulse (s)' to numeric
df['specific impulse (s)'] = pd.to_numeric(df['specific impulse (s)'])
# Filter engines with specific impulse > 1000
filtered_engines = df[df['specific impulse (s)'] > 1000]
# Count the number of such engines
count = len(filtered_engines)
print(f"Final Answer: {count}")
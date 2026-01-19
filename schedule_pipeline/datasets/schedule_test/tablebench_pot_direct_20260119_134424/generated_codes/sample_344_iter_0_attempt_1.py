import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'specific impulse (s)' to numeric
df['specific impulse (s)'] = pd.to_numeric(df['specific impulse (s)'])
# Count engines with specific impulse > 1000
count = df[df['specific impulse (s)'] > 1000].shape[0]
print(f"Final Answer: {count}")
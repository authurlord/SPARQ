import pandas as pd

df = pd.read_csv('table.csv')
# Convert the residential monthly usage column to numeric
df['residential monthly usage : 1000 kwh'] = pd.to_numeric(df['residential monthly usage : 1000 kwh'])
# Count cities with usage above 12
count_above_12 = (df['residential monthly usage : 1000 kwh'] > 12).sum()
print(f"Final Answer: {count_above_12}")
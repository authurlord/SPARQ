import pandas as pd

df = pd.read_csv('table.csv')
# Convert the residential monthly usage column to float
residential_usage = df['residential monthly usage : 1000 kwh'].astype(float)
# Count cities with usage above 12
count_above_12 = (residential_usage > 12).sum()
print(f"Final Answer: {count_above_12}")
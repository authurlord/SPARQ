import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'residential monthly usage : 1000 kwh' to float and filter values > 12
count_cities = df['residential monthly usage : 1000 kwh'].astype(float).gt(12).sum()
print(f"Final Answer: {count_cities}")
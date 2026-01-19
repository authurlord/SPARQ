import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'residential monthly usage : 1000 kwh' to float and filter values above 12
above_12 = df['residential monthly usage : 1000 kwh'].astype(float) > 12
count_cities = above_12.sum()
print(f"Final Answer: {count_cities}")
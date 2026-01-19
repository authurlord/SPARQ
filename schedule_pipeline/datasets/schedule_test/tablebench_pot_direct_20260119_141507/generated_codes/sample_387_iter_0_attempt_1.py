import pandas as pd

df = pd.read_csv('table.csv')
# Filter cities where residential monthly usage is above 12 (in 1000 kwh)
above_12 = df['residential monthly usage : 1000 kwh'].astype(float) > 12
count_above_12 = above_12.sum()
print(f"Final Answer: {count_above_12}")
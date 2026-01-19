import pandas as pd

df = pd.read_csv('table.csv')
# Filter cities where residential monthly usage > 12
count_above_12 = df[df['residential monthly usage : 1000 kwh'] > 12].shape[0]
print(f"Final Answer: {count_above_12}")
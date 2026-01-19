import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'residential monthly usage : 1000 kwh' to float and count values > 12
count_above_12 = (df['residential monthly usage : 1000 kwh'].astype(float) > 12).sum()
print(f"Final Answer: {count_above_12}")
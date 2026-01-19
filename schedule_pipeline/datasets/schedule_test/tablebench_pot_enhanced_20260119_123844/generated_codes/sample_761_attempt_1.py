import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the average number of bills originally cosponsored
forecasted_bills = df['bills originally cosponsored'].mean()
print(f"Final Answer: {forecasted_bills:.1f}")
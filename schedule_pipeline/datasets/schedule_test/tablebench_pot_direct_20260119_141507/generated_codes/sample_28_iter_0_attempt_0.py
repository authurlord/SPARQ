import pandas as pd

df = pd.read_csv('table.csv')
# Select top 5 ranked governors (rank <= 5)
top_5 = df[df['Rank'].le(5)]
# Calculate the average number of days served
avg_days = top_5['Days'].mean()
print(f"Final Answer: {avg_days:.0f}")
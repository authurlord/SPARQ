import pandas as pd

df = pd.read_csv('table.csv')
# Filter top 5 ranked governors
top_5 = df[df['Rank'].astype(int) <= 5]
# Convert 'Days' to integer and calculate average
avg_days = top_5['Days'].astype(int).mean()
print(f"Final Answer: {avg_days:.1f}")
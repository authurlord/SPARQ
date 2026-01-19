import pandas as pd

df = pd.read_csv('table.csv')
# Filter top 5 ranked governors
top_5 = df[df['Rank'].astype(int) <= 5]
# Convert 'Days' to integer
top_5['Days'] = top_5['Days'].astype(int)
# Calculate average days
avg_days = top_5['Days'].mean()
print(f"Final Answer: {avg_days:.1f}")
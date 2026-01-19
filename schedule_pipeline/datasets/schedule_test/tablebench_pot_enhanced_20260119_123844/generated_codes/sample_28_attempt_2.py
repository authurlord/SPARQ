import pandas as pd

df = pd.read_csv('table.csv')
# Filter top 5 ranked governors (Rank 1 to 5)
top_5 = df[df['Rank'].astype(int) <= 5]
# Convert 'Days' to integer and calculate average
average_days = top_5['Days'].astype(int).mean()
print(f"Final Answer: {average_days:.1f}")
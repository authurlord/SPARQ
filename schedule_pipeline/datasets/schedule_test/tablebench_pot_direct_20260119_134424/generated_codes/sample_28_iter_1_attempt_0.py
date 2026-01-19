import pandas as pd

df = pd.read_csv('table.csv')
# Clean the 'Days' column by removing commas and converting to integer
df['Days'] = df['Days'].str.replace(',', '').astype(int)
# Filter top 5 ranked governors (rank 1 to 5)
top_5 = df[df['Rank'].astype(int) <= 5]
# Calculate average days served
average_days = top_5['Days'].mean()
print(f"Final Answer: {average_days:.1f}")
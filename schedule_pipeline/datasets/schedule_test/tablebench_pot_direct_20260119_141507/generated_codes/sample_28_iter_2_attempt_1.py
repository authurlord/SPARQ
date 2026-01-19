import pandas as pd

df = pd.read_csv('table.csv')
# Filter top 5 ranked governors (rank 1 to 5)
top_5 = df[df['Rank'].isin([1, 2, 3, 4, 5])]
# Clean the 'Days' column by removing commas and converting to integer
top_5['Days'] = top_5['Days'].str.replace(',', '').astype(int)
# Calculate the average number of days
average_days = top_5['Days'].mean()
print(f"Final Answer: {average_days:.0f}")
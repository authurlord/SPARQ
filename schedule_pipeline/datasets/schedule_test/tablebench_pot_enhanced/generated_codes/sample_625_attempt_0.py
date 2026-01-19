import pandas as pd

df = pd.read_csv('table.csv')
# Select top 3 metro areas based on rank (wjc)
top_3 = df.head(3)
# Convert 'number of jews (wjc)' to integer and sum
total_jews = top_3['number of jews (wjc)'].astype(int).sum()
# Calculate average
average_jews = total_jews / 3
print(f"Final Answer: {average_jews:.0f}")
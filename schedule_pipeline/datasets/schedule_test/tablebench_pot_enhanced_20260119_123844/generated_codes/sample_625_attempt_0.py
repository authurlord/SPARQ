import pandas as pd

df = pd.read_csv('table.csv')
# Select top 3 metro areas based on rank (wjc)
top_3 = df.head(3)
# Convert 'number of jews (wjc)' to integer for calculation
top_3['number of jews (wjc)'] = top_3['number of jews (wjc)'].astype(int)
# Calculate average
average_jews = top_3['number of jews (wjc)'].mean()
print(f"Final Answer: {average_jews:.0f}")
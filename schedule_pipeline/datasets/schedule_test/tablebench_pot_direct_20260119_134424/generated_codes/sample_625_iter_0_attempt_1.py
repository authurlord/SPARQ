import pandas as pd

df = pd.read_csv('table.csv')
# Select the top 3 metro areas based on rank (wjc)
top_3_jews = df['number of jews (wjc)'].head(3).astype(int)
# Calculate average
average_jews = top_3_jews.sum() / 3
print(f"Final Answer: {average_jews:.0f}")
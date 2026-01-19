import pandas as pd

df = pd.read_csv('table.csv')
# Filter top 3 metro areas by rank (wjc): 1, 2, 3
top_3 = df[df['rank ( wjc )'].isin(['1', '2', '3'])]
# Extract number of Jews (asarb) for these areas
jews_asarb = top_3['number of jews (asarb)'].astype(int)
total_jews = jews_asarb.sum()
average_jews = total_jews / 3
print(f"Final Answer: {average_jews:.0f}")
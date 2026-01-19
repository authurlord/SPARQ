import pandas as pd

df = pd.read_csv('table.csv')
# Extract total medals for top 5 nations (rank 1 to 5)
top_5_total = df['total'].head(5)
median_top_5 = top_5_total.median()

# Extract total medals for all nations
all_total = df['total']
median_all = all_total.median()

print(f"Final Answer: {median_top_5}, {median_all}")
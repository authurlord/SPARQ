import pandas as pd

df = pd.read_csv('table.csv')
# Sum the 'hk viewers' for the top 5 shows (rank 1 to 5)
total_viewership = df.head(5)['hk viewers'].sum()
print(f"Final Answer: {total_viewership}")
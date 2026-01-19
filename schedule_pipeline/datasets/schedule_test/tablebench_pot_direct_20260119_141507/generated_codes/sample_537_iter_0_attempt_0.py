import pandas as pd

df = pd.read_csv('table.csv')
# Sort by year to process in chronological order
df_sorted = df.sort_values(by='year (january)')
# Find the first year where urbanization rate > 50%
for index, row in df_sorted.iterrows():
    if row['urban , %'] > 50:
        print(f"Final Answer: {row['year (january)']}")
        break
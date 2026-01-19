import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where the country has "(2001)" in the name
filtered_df = df[df['country'].str.endswith('(2001)')]
# Get the maximum value in 'orphans as % of all children' and its corresponding country
max_orphan_percentage = filtered_df.loc[filtered_df['orphans as % of all children'].idxmax(), 'country']
print(f"Final Answer: {max_orphan_percentage}")
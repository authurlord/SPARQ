import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for 2008 and 2009
filtered_df = df[df['year'].isin(['2008', '2009'])]
# Sum the wins
total_wins = filtered_df['wins'].sum()
print(f"Final Answer: {total_wins}")
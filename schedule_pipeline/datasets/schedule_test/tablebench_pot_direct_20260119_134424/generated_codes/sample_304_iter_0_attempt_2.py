import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where venue is 'waverley park' and crowd > 40000
filtered_df = df[(df['venue'] == 'waverley park') & (df['crowd'].astype(int) > 40000)]
# Find the row with the highest margin
max_margin_row = filtered_df.loc[filtered_df['margin'].idxmax()]
# Get the winner team
winner_team = max_margin_row['winners']
print(f"Final Answer: {winner_team}")
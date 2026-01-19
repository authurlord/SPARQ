import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where venue is 'waverley park' and crowd > 40000
filtered_df = df[(df['venue'] == 'waverley park') & (df['crowd'].astype(int) > 40000)]
# Find the team with the highest margin
max_margin_team = filtered_df.loc[filtered_df['margin'].idxmax(), 'winners']
print(f"Final Answer: {max_margin_team}")
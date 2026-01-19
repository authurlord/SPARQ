import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where venue is Waverley Park and crowd > 40,000
filtered_df = df[(df['venue'] == 'waverley park') & (df['crowd'].astype(int) > 40000)]
# Find the team (winners) with the highest margin
highest_margin_team = filtered_df.loc[filtered_df['margin'].idxmax(), 'winners']
print(f"Final Answer: {highest_margin_team}")
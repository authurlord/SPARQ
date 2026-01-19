import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where venue is Waverley Park and crowd > 40000
filtered_df = df[(df['venue'] == 'waverley park') & (df['crowd'] > 40000)]
# Find the team (winners) with the highest margin
max_margin_row = filtered_df.loc[filtered_df['margin'].idxmax()]
final_team = max_margin_row['winners']
print(f"Final Answer: {final_team}")
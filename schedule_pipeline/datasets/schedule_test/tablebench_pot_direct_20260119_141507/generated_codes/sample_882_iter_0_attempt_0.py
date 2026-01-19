import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Liverpool and Premier League
filtered_df = df[(df['Club'] == 'Liverpool') & (df['League'] == 'Premier League')]
# Get the season with maximum goals in League_1 (Premier League goals)
max_goals_row = filtered_df.loc[filtered_df['League_1'].idxmax()]
final_season = max_goals_row['Season']
print(f"Final Answer: {final_season}")
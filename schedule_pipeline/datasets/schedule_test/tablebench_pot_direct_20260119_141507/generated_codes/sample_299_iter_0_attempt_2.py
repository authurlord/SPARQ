import pandas as pd

df = pd.read_csv('table.csv')
# Filter for World Championships with 800 m event
filtered_df = df[(df['Competition'] == 'World Championships') & (df['Event'] == '800 m')]
# Find the row with the best (lowest) position
best_position_row = filtered_df.loc[filtered_df['Position'].astype(int).idxmin()]
# Extract the year
best_year = best_position_row['Year']
print(f"Final Answer: {best_year}")
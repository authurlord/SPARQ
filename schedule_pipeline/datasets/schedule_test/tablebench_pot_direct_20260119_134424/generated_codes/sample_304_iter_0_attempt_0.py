import pandas as pd

df = pd.read_csv('table.csv')
# Convert crowd to integer for comparison
df['crowd'] = pd.to_numeric(df['crowd'], errors='coerce')
# Filter rows where venue is 'waverley park' and crowd > 40000
filtered_df = df[(df['venue'] == 'waverley park') & (df['crowd'] > 40000)]
# Find the team with the highest margin
winner_with_highest_margin = filtered_df.loc[filtered_df['margin'].idxmax(), 'winners']
print(f"Final Answer: {winner_with_highest_margin}")
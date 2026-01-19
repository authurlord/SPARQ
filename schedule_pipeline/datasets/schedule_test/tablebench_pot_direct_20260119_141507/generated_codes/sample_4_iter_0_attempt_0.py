import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Club is 'Liverpool' and League is 'Premier League'
filtered_df = df[(df['Club'] == 'Liverpool') & (df['League'] == 'Premier League')]
# Sum the 'Goals' column from the filtered rows
total_goals = filtered_df['Goals'].sum()
print(f"Final Answer: {total_goals}")
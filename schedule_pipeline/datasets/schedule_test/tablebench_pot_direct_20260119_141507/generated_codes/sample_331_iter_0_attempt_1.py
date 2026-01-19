import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Event is '100 m' and Competition is 'African Junior Championships'
filtered_df = df[(df['Event'] == '100 m') & (df['Competition'] == 'African Junior Championships')]
# Check for 1st position (gold medal)
gold_medal_year = filtered_df[filtered_df['Position'] == '1st']['Year'].values
# Since only one such row exists, return the year
print(f"Final Answer: {gold_medal_year[0]}")
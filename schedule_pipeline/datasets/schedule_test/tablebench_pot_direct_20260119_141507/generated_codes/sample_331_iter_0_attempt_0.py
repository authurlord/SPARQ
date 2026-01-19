import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Event is '100 m' and Competition is 'African Junior Championships'
filtered_rows = df[(df['Event'] == '100 m') & (df['Competition'] == 'African Junior Championships')]
# Find the row with '1st' position
gold_medal_row = filtered_rows[filtered_rows['Position'] == '1st']
# Extract the year
year = gold_medal_row.iloc[0]['Year']
print(f"Final Answer: {year}")
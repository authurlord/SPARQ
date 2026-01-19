import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 100m event, African Junior Championships, and 1st position
gold_medal_row = df[(df['Event'] == '100 m') & 
                    (df['Competition'] == 'African Junior Championships') & 
                    (df['Position'] == '1st')]
# Extract the year
year = gold_medal_row['Year'].iloc[0]
print(f"Final Answer: {year}")
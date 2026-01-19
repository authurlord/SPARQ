import pandas as pd

df = pd.read_csv('table.csv')
# Filter for gold medal in 100m at African Junior Championships
gold_medal_100m = df[(df['Event'] == '100 m') & 
                     (df['Competition'] == 'African Junior Championships') & 
                     (df['Position'] == '1st')]
# Extract the year
year = gold_medal_100m['Year'].iloc[0]
print(f"Final Answer: {year}")
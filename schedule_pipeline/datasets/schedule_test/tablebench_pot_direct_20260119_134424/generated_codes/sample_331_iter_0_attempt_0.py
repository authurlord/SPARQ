import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 100 m event, African Junior Championships, and 1st position
gold_medal_year = df[(df['Event'] == '100 m') & 
                     (df['Competition'] == 'African Junior Championships') & 
                     (df['Position'] == '1st')]['Year'].iloc[0]
print(f"Final Answer: {gold_medal_year}")
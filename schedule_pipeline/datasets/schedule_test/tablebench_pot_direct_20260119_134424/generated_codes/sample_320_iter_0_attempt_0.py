import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 20 km events where the athlete finished 1st (gold medal)
gold_medal_year = df[(df['Event'] == '20 km') & (df['Position'] == '1st')]['Year'].iloc[0]
print(f"Final Answer: {gold_medal_year}")
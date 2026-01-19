import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 1st position in 20 km events
gold_medal_years = df[(df['Position'] == '1st') & (df['Event'] == '20 km')]['Year']
# Since there might be multiple entries, we return all years
print(f"Final Answer: {', '.join(gold_medal_years.tolist())}")
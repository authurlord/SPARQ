import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 20 km events with gold medal (GR in Notes)
gold_medal_row = df[(df['Event'] == '20 km') & (df['Notes'].str.contains('GR', na=False))]
# Extract the year
year_gold_medal = gold_medal_row['Year'].iloc[0]
print(f"Final Answer: {year_gold_medal}")
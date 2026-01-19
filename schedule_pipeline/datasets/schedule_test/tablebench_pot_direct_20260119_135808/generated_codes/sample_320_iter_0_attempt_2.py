import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 20 km events with gold medal (GR in Notes)
gold_medal_year = df[(df['Event'] == '20 km') & (df['Notes'].str.contains('GR', na=False))]['Year'].iloc[0]
print(f"Final Answer: {gold_medal_year}")
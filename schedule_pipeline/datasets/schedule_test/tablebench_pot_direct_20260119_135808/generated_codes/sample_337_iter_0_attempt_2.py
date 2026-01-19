import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Asian Games, 200m event, and PB (personal best)
filtered_df = df[(df['Competition'] == 'Asian Games') & 
                  (df['Event'] == '200 m') & 
                  (df['Position'].str.contains('PB', na=False))]
# Extract the year
year_pb = filtered_df['Year'].iloc[0]
print(f"Final Answer: {year_pb}")
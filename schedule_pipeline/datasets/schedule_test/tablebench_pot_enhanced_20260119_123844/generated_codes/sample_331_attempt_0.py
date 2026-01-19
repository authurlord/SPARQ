import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 100 m event, African Junior Championships, and 1st position
filtered_df = df[(df['Event'] == '100 m') & 
                 (df['Competition'] == 'African Junior Championships') & 
                 (df['Position'] == '1st')]
# Extract the year
year = filtered_df['Year'].iloc[0]
print(f"Final Answer: {year}")
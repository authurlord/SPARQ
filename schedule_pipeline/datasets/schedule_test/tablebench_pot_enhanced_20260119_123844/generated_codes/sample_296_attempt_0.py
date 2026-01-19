import pandas as pd

df = pd.read_csv('table.csv')
# Filter for the specific conditions
filtered_df = df[(df['Event'] == '200 m') & 
                  (df['Competition'] == 'European Junior Championships') & 
                  (df['Position'] == '1st')]
# Extract the year
year = filtered_df['Year'].iloc[0]
print(f"Final Answer: {year}")
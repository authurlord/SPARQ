import pandas as pd

df = pd.read_csv('table.csv')
# Filter for European Junior Championships, 200m event, and 1st position
filtered_df = df[(df['Competition'] == 'European Junior Championships') & 
                 (df['Event'] == '200 m') & 
                 (df['Position'] == '1st')]

# Get the year from the filtered result
year = filtered_df['Year'].iloc[0]
print(f"Final Answer: {year}")
import pandas as pd

df = pd.read_csv('table.csv')
# Filter for European Junior Championships, 200m event, and 1st position
filtered_data = df[(df['Competition'] == 'European Junior Championships') & 
                   (df['Event'] == '200 m') & 
                   (df['Position'] == '1st')]

# Extract the year
year = filtered_data['Year'].iloc[0]
print(f"Final Answer: {year}")
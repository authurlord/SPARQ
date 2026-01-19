import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 100m event, 1st position, and African Junior Championships
filtered_df = df[(df['Event'] == '100 m') & (df['Position'] == '1st') & (df['Competition'] == 'African Junior Championships')]
# Extract the year
year = filtered_df['Year'].iloc[0]
print(f"Final Answer: {year}")
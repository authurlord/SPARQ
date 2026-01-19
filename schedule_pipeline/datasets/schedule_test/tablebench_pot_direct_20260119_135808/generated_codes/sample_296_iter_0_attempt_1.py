import pandas as pd

df = pd.read_csv('table.csv')
# Filter for gold medal in 200m at European Junior Championships
filtered_df = df[(df['Event'] == '200 m') & (df['Position'] == '1st') & (df['Competition'] == 'European Junior Championships')]
year = filtered_df['Year'].iloc[0]
print(f"Final Answer: {year}")
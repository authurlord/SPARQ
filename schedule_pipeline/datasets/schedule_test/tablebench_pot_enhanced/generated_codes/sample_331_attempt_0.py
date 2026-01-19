import pandas as pd

df = pd.read_csv('table.csv')
# Filter for gold medal in 100m at African Junior Championships
filtered_df = df[(df['Event'] == '100 m') & (df['Position'] == '1st') & (df['Competition'] == 'African Junior Championships')]
year = filtered_df['Year'].iloc[0]
print(f"Final Answer: {year}")
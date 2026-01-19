import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for African Championships and 20 km walk
african_championships_20km = df[(df['Competition'] == 'African Championships') & (df['Event'] == '20 km walk')]

# Find the row with the best time (lowest time in Notes)
# The personal best is marked as (CR), so look for that
best_time_row = african_championships_20km[african_championships_20km['Notes'].str.contains('CR', na=False)]

# Extract the year
year_best_time = best_time_row['Year'].values[0]
print(f"Final Answer: {year_best_time}")
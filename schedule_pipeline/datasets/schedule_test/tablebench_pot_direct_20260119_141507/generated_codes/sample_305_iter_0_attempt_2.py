import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Event is '400 m'
filtered_df = df[df['Event'] == '400 m']
# Convert 'Notes' to float to extract times and find the minimum
times = filtered_df['Notes'].astype(float)
# Get the row with the minimum time
best_time_row = filtered_df.iloc[times.idxmin()]
# Extract the year from the row
year = best_time_row['Year']
print(f"Final Answer: {year}")
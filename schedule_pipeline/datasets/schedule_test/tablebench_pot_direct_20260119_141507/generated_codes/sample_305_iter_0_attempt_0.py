import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where the event is "400 m"
filtered_df = df[df['Event'] == '400 m']
# Convert Notes to float for comparison and find the minimum time
best_time = filtered_df['Notes'].astype(float).min()
# Find the corresponding year where this best time occurred
best_year = filtered_df[filtered_df['Notes'] == str(best_time)].iloc[0]['Year']
print(f"Final Answer: {best_year}")
import pandas as pd

df = pd.read_csv('table.csv')
# Filter for European Championships and 400 m event
filtered_df = df[(df['Competition'] == 'European Championships') & (df['Event'] == '400 m')]
# Convert position to numeric (e.g., '7th' -> 7)
filtered_df['Position'] = filtered_df['Position'].str.extract(r'(\d+)')
filtered_df['Position'] = pd.to_numeric(filtered_df['Position'], errors='coerce')
# Find the row with the lowest position (best result)
best_position_row = filtered_df.loc[filtered_df['Position'].idxmin()]
best_year = best_position_row['Year']
print(f"Final Answer: {best_year}")
import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 400m events at European Championships
filtered_df = df[(df['Event'] == '400 m') & (df['Competition'] == 'European Championships')]

# Convert Position to numeric, treating non-numeric values as NaN
filtered_df['Position'] = pd.to_numeric(filtered_df['Position'], errors='coerce')

# Drop rows where Position is NaN (like '–', '17th (sf)')
valid_positions = filtered_df.dropna(subset=['Position'])

# Find the row with the best (lowest) position
best_position_row = valid_positions.loc[valid_positions['Position'].idxmin()]

# Extract the year
best_year = best_position_row['Year']
print(f"Final Answer: {best_year}")
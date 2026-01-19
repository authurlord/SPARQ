import pandas as pd

df = pd.read_csv('table.csv')

# Filter for European Championships and 400m event
filtered_df = df[(df['Competition'] == 'European Championships') & (df['Event'] == '400 m')]

# Convert Position to numeric, invalid entries become NaN
filtered_df['Position'] = pd.to_numeric(filtered_df['Position'], errors='coerce')

# Drop rows with invalid positions
valid_positions = filtered_df.dropna(subset=['Position'])

# Find the row with the best (lowest) position
best_row = valid_positions.loc[valid_positions['Position'].idxmin()]

# Get the year
best_year = best_row['Year']
print(f"Final Answer: {best_year}")
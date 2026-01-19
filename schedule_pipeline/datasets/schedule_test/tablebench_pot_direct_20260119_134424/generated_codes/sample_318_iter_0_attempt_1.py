import pandas as pd

df = pd.read_csv('table.csv')

# Filter for European Championships and 400m event
filtered_df = df[(df['Competition'] == 'European Championships') & (df['Event'] == '400 m')]

# Convert Position to numeric, coercing non-numeric values to NaN
filtered_df['Position'] = pd.to_numeric(filtered_df['Position'], errors='coerce')

# Drop rows with NaN in Position (like '–' or '17th (sf)')
valid_positions = filtered_df.dropna(subset=['Position'])

# Find the minimum (best) position
best_position = valid_positions['Position'].min()

# Get the year corresponding to the best position
best_year = valid_positions[valid_positions['Position'] == best_position]['Year'].iloc[0]

print(f"Final Answer: {best_year}")
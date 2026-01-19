import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Remove the "totals" row
df = df[df['manager'] != 'totals']

# Convert 'wins' and 'games' to numeric, coercing errors to NaN if any
df['wins'] = pd.to_numeric(df['wins'], errors='coerce')
df['games'] = pd.to_numeric(df['games'], errors='coerce')

# Filter out any rows where games is 0 or NaN (to avoid division by zero)
df = df[(df['games'] > 0) & (df['wins'].notna()) & (df['games'].notna())]

# Calculate winning percentage
df['win_percentage'] = df['wins'] / df['games']

# Find the manager with the highest winning percentage
best_manager = df.loc[df['win_percentage'].idxmax(), 'manager']

print(f"Final Answer: {best_manager}")
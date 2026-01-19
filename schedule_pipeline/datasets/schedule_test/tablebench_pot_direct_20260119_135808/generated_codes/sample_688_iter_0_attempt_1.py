import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'wins' and 'games' to numeric, skip 'totals'
df['wins'] = pd.to_numeric(df['wins'], errors='coerce')
df['games'] = pd.to_numeric(df['games'], errors='coerce')

# Filter out the 'totals' row
df = df[df['manager'] != 'totals']

# Calculate winning percentage
df['win_percentage'] = df['wins'] / df['games']

# Find the manager with the highest winning percentage
best_manager = df.loc[df['win_percentage'].idxmax(), 'manager']

print(f"Final Answer: {best_manager}")
import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'wins' and 'games' to numeric, handling non-numeric values
df['wins'] = pd.to_numeric(df['wins'], errors='coerce')
df['games'] = pd.to_numeric(df['games'], errors='coerce')

# Calculate winning percentage
df['winning_percentage'] = df['wins'] / df['games']

# Exclude the 'totals' row
df = df[df['manager'] != 'totals']

# Find the manager with the highest winning percentage
max_win_pct_manager = df.loc[df['winning_percentage'].idxmax(), 'manager']

print(f"Final Answer: {max_win_pct_manager}")
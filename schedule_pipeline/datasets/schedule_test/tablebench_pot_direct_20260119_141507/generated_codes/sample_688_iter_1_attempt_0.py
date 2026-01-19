import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'wins' and 'games' to numeric, handling any potential parsing issues
df[['wins', 'games']] = df[['wins', 'games']].apply(pd.to_numeric, errors='coerce')

# Remove the 'totals' row
df = df[df['manager'] != 'totals']

# Calculate winning percentage
df['win_percentage'] = df['wins'] / df['games']

# Find the manager with the highest winning percentage
best_manager = df.loc[df['win_percentage'].idxmax(), 'manager']
print(f"Final Answer: {best_manager}")
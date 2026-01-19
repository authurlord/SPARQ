import pandas as pd

df = pd.read_csv('table.csv')
# Remove the 'totals' row
df = df[df['manager'] != 'totals']

# Convert 'wins' and 'games' to numeric, handling any possible non-numeric strings
df['wins'] = pd.to_numeric(df['wins'], errors='coerce')
df['games'] = pd.to_numeric(df['games'], errors='coerce')

# Drop rows where either wins or games is NaN (invalid)
df = df.dropna(subset=['wins', 'games'])

# Calculate winning percentage
df['win_percentage'] = df['wins'] / df['games']

# Find the manager with the highest winning percentage
best_manager = df.loc[df['win_percentage'].idxmax(), 'manager']
print(f"Final Answer: {best_manager}")
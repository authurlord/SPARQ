import pandas as pd

df = pd.read_csv('table.csv')
# Remove the 'totals' row
df = df[df['manager'] != 'totals']

# Calculate winning percentage (wins / games)
df['win_percentage'] = df['wins'] / df['games']

# Find the manager with the highest winning percentage
max_percentage_manager = df.loc[df['win_percentage'].idxmax(), 'manager']
print(f"Final Answer: {max_percentage_manager}")
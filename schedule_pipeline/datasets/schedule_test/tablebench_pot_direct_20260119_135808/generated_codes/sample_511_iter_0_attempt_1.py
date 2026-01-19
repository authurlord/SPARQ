import pandas as pd

df = pd.read_csv('table.csv')
# Calculate Dance Efficiency
df['Dance Efficiency'] = df['total points'].astype(int) / df['number of dances'].astype(int)
# Find the couple with the highest Dance Efficiency
max_efficiency_couple = df.loc[df['Dance Efficiency'].idxmax(), 'couple']
print(f"Final Answer: {max_efficiency_couple}")
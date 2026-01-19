import pandas as pd

df = pd.read_csv('table.csv')
# Calculate Dance Efficiency: total points / number of dances
df['dance_efficiency'] = df['total points'].astype(float) / df['number of dances'].astype(float)
# Find the couple with the highest dance efficiency
highest_efficiency_couple = df.loc[df['dance_efficiency'].idxmax(), 'couple']
print(f"Final Answer: {highest_efficiency_couple}")
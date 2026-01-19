import pandas as pd

df = pd.read_csv('table.csv')
# Find the population of the 6th ranked city ('valenza') and 'acqui terme'
valenza_pop = df[df['city'] == 'valenza']['population'].values[0]
acqui_terme_pop = df[df['city'] == 'acqui terme']['population'].values[0]

# Calculate the difference
difference = valenza_pop - acqui_terme_pop
print(f"Final Answer: {difference}")
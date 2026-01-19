import pandas as pd

df = pd.read_csv('table.csv')
# Get population of the 6th ranked city (valenza)
population_6th = df[df['rank'] == '6th']['population'].values[0]
# Get population of acqui terme
population_acqui = df[df['city'] == 'acqui terme']['population'].values[0]
# Calculate the difference
difference = abs(int(population_6th) - int(population_acqui))
print(f"Final Answer: {difference}")
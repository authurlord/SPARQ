import pandas as pd

df = pd.read_csv('table.csv')
# Get population of 6th ranked city (valenza)
pop_valenza = df[df['rank'] == '6th']['population'].values[0]
# Get population of acqui terme
pop_acqui = df[df['city'] == 'acqui terme']['population'].values[0]
# Calculate difference
difference = abs(int(pop_valenza) - int(pop_acqui))
print(f"Final Answer: {difference}")
import pandas as pd

df = pd.read_csv('table.csv')
# Filter for valenza (6th ranked city)
valenza_population = df[df['city'] == 'valenza']['population'].values[0]
# Filter for acqui terme
acqui_terme_population = df[df['city'] == 'acqui terme']['population'].values[0]
# Calculate difference
difference = int(valenza_population) - int(acqui_terme_population)
print(f"Final Answer: {difference}")
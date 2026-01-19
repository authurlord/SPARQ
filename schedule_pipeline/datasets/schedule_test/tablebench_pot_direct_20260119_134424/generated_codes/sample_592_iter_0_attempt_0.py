import pandas as pd

df = pd.read_csv('table.csv')
# Get population of 'acqui terme'
acqui_population = df[df['city'] == 'acqui terme']['population'].values[0]
# Get population of 'valenza' (6th ranked city)
valenza_population = df[df['rank'] == '6th']['population'].values[0]
# Calculate the difference
difference = int(valenza_population) - int(acqui_population)
print(f"Final Answer: {difference}")
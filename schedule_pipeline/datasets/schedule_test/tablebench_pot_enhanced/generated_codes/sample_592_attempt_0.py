import pandas as pd

df = pd.read_csv('table.csv')
# Extract population for 'valenza' (6th ranked city)
valenza_population = df[df['city'] == 'valenza']['population'].values[0]
# Extract population for 'acqui terme'
acqui_terme_population = df[df['city'] == 'acqui terme']['population'].values[0]
# Calculate the difference
difference = int(valenza_population) - int(acqui_terme_population)
print(f"Final Answer: {difference}")
import pandas as pd

df = pd.read_csv('table.csv')
# Find population of the 6th ranked city
sixth_city_population = df[df['rank'] == '6th']['population'].values[0]
# Find population of 'acqui terme'
acqui_terme_population = df[df['city'] == 'acqui terme']['population'].values[0]
# Calculate the difference
difference = int(sixth_city_population) - int(acqui_terme_population)
print(f"Final Answer: {difference}")
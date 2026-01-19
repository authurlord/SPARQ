import pandas as pd

df = pd.read_csv('table.csv')
# Find reported offenses for 'larceny - theft' and 'burglary'
larceny_theft = df[df['crime'] == 'larceny - theft']['reported offenses'].values[0]
burglary = df[df['crime'] == 'burglary']['reported offenses'].values[0]
difference = larceny_theft - burglary
print(f"Final Answer: {difference}")
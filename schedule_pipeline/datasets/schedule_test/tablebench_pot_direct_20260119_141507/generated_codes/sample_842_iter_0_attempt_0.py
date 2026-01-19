import pandas as pd

df = pd.read_csv('table.csv')
# Find the reported offenses for 'larceny - theft' and 'burglary'
larceny_theft_offenses = df[df['crime'] == 'larceny - theft']['reported offenses'].values[0]
burglary_offenses = df[df['crime'] == 'burglary']['reported offenses'].values[0]
difference = larceny_theft_offenses - burglary_offenses
print(f"Final Answer: {difference}")
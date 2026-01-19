import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for 'larceny - theft' and 'burglary'
larceny_offenses = df[df['crime'] == 'larceny - theft']['reported offenses'].values[0]
burglary_offenses = df[df['crime'] == 'burglary']['reported offenses'].values[0]

# Calculate the difference
difference = int(larceny_offenses) - int(burglary_offenses)
print(f"Final Answer: {difference}")
import pandas as pd

df = pd.read_csv('table.csv')
# Extract reported offenses for larceny-theft and burglary
larceny_offenses = int(df[df['crime'] == 'larceny - theft']['reported offenses'].iloc[0])
burglary_offenses = int(df[df['crime'] == 'burglary']['reported offenses'].iloc[0])
# Calculate the difference
difference = larceny_offenses - burglary_offenses
print(f"Final Answer: {difference}")
import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'reported offenses' column to numeric
df['reported offenses'] = pd.to_numeric(df['reported offenses'], errors='coerce')

# Get reported offenses for 'larceny - theft' and 'burglary'
larceny_theft = df[df['crime'] == 'larceny - theft']['reported offenses'].values[0]
burglary = df[df['crime'] == 'burglary']['reported offenses'].values[0]

# Calculate the difference
difference = larceny_theft - burglary
print(f"Final Answer: {difference}")
import pandas as pd

df = pd.read_csv('table.csv')

# Extract male population data
males_10_to_29 = df.loc[df['SPECIFICATION'] == 'I.1.A.', ['10–19', '20–29']].sum().sum()
males_60_plus = df.loc[df['SPECIFICATION'] == 'I.1.A.', ['60–69', '70–79', '80 +']].sum().sum()

# Calculate the difference
difference = males_10_to_29 - males_60_plus

print(f"Final Answer: {difference}")
import pandas as pd

df = pd.read_csv('table.csv')

# Extract male population for 10-29 age group (10–19 and 20–29)
males_10_29 = df.loc[df['SPECIFICATION'] == 'I.1.A.', 'POPULATION (by age group in 2002)_2'].iloc[0] + \
               df.loc[df['SPECIFICATION'] == 'I.1.A.', 'POPULATION (by age group in 2002)_3'].iloc[0]

# Extract male population for 60+ age group (60–69, 70–79, 80+)
males_60_plus = df.loc[df['SPECIFICATION'] == 'I.1.A.', 'POPULATION (by age group in 2002)_6'].iloc[0] + \
                 df.loc[df['SPECIFICATION'] == 'I.1.A.', 'POPULATION (by age group in 2002)_7'].iloc[0] + \
                 df.loc[df['SPECIFICATION'] == 'I.1.A.', 'POPULATION (by age group in 2002)_8'].iloc[0]

# Calculate the difference
difference = males_10_29 - males_60_plus

print(f"Final Answer: {difference}")
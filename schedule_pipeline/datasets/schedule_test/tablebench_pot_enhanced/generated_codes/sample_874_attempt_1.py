import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract male population values for relevant age groups
males_10_19 = df.loc[df['SPECIFICATION'] == 'I.', 'POPULATION (by age group in 2002)_2'].iloc[4]
males_20_29 = df.loc[df['SPECIFICATION'] == 'I.', 'POPULATION (by age group in 2002)_3'].iloc[4]
males_60_plus = df.loc[df['SPECIFICATION'] == 'I.', 'POPULATION (by age group in 2002)_6'].iloc[4] + \
                 df.loc[df['SPECIFICATION'] == 'I.', 'POPULATION (by age group in 2002)_7'].iloc[4] + \
                 df.loc[df['SPECIFICATION'] == 'I.', 'POPULATION (by age group in 2002)_8'].iloc[4]

# Calculate total males in 10–29 and 60+
males_10_29 = int(males_10_19) + int(males_20_29)
males_60_plus = int(males_60_plus)

# Compute difference
difference = males_10_29 - males_60_plus

print(f"Final Answer: {difference}")
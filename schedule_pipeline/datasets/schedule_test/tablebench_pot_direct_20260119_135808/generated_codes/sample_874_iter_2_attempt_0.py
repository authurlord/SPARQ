import pandas as pd

df = pd.read_csv('table.csv')

# Extract male population data
males_10_19 = df.loc[df['SPECIFICATION'] == 'I.', 'POPULATION (by age group in 2002)_2'].values[0]
males_20_29 = df.loc[df['SPECIFICATION'] == 'I.', 'POPULATION (by age group in 2002)_3'].values[0]
males_60_plus = df.loc[df['SPECIFICATION'] == 'I.', 'POPULATION (by age group in 2002)_6'].values[0] + \
                df.loc[df['SPECIFICATION'] == 'I.', 'POPULATION (by age group in 2002)_7'].values[0] + \
                df.loc[df['SPECIFICATION'] == 'I.', 'POPULATION (by age group in 2002)_8'].values[0]

# Calculate total males in 10–29 and 60+
total_males_10_29 = males_10_19 + males_20_29
total_males_60_plus = males_60_plus

# Compute the difference
difference = total_males_10_29 - total_males_60_plus

print(f"Final Answer: {difference}")
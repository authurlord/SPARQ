import pandas as pd

df = pd.read_csv('table.csv')
# Extract male population for 10-29 age group (10-19 and 20-29)
males_10_29 = df.iloc[5]['POPULATION (by age group in 2002)_2'] + df.iloc[5]['POPULATION (by age group in 2002)_3']
# Extract male population for 60+ age group (60-69 and 80+)
males_60_plus = df.iloc[5]['POPULATION (by age group in 2002)_6'] + df.iloc[5]['POPULATION (by age group in 2002)_9']
# Calculate the difference
difference = males_10_29 - males_60_plus
print(f"Final Answer: {difference}")
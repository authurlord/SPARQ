import pandas as pd

df = pd.read_csv('table.csv')

# Extract male population for age groups
males_10_19 = df.iloc[4]['POPULATION (by age group in 2002)_2']  # 10–19
males_20_29 = df.iloc[4]['POPULATION (by age group in 2002)_3']  # 20–29
males_60_69 = df.iloc[4]['POPULATION (by age group in 2002)_6']  # 60–69
males_70_79 = df.iloc[4]['POPULATION (by age group in 2002)_7']  # 70–79
males_80_plus = df.iloc[4]['POPULATION (by age group in 2002)_8']  # 80+

# Calculate total males in 10–29 and 60+
males_10_29 = males_10_19 + males_20_29
males_60_plus = males_60_69 + males_70_79 + males_80_plus

# Calculate the difference
difference = males_10_29 - males_60_plus
print(f"Final Answer: {difference}")
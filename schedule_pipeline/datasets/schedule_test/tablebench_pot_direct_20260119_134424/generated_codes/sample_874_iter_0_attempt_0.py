import pandas as pd

df = pd.read_csv('table.csv')

# Extract the row for males (row index 4)
males_10_19 = df.iloc[4]['POPULATION (by age group in 2002)_2']  # 10–19
males_20_29 = df.iloc[4]['POPULATION (by age group in 2002)_3']  # 20–29
males_60_69 = df.iloc[4]['POPULATION (by age group in 2002)_6']  # 60–69
males_80_plus = df.iloc[4]['POPULATION (by age group in 2002)_9']  # 80+

# Calculate totals
males_10_29 = males_10_19 + males_20_29
males_60_plus = males_60_69 + males_80_plus

# Calculate difference
difference = males_10_29 - males_60_plus
print(f"Final Answer: {difference}")
import pandas as pd

df = pd.read_csv('table.csv')
# Get civil liberties scores for 1972 and 1980
civil_liberties_1972 = df.loc[df['year'] == '1972', 'civil liberties'].values[0]
civil_liberties_1980 = df.loc[df['year'] == '1980', 'civil liberties'].values[0]

# Calculate percentage change
percentage_change = ((civil_liberties_1980 - civil_liberties_1972) / civil_liberties_1972) * 100

print(f"Final Answer: {percentage_change:.1f}")
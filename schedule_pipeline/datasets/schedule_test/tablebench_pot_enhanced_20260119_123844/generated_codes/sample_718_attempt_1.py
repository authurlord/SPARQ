import pandas as pd

df = pd.read_csv('table.csv')
# Find the university with the highest research funding
max_funding = df['research funding (000)'].max()
min_funding = df['research funding (000)'].min()
difference = max_funding - min_funding
# Find the institution with the highest research funding
highest_institution = df.loc[df['research funding (000)'] == max_funding, 'institution'].values[0]
print(f"Final Answer: {highest_institution}, {difference}")
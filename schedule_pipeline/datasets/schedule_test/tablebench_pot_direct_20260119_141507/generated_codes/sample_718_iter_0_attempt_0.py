import pandas as pd

df = pd.read_csv('table.csv')
# Find the university with highest and lowest research funding
max_funding = df.loc[df['research funding (000)'].idxmax(), 'institution']
min_funding = df.loc[df['research funding (000)'].idxmin(), 'institution']
max_value = df['research funding (000)'].max()
min_value = df['research funding (000)'].min()
difference = max_value - min_value

print(f"Final Answer: {max_funding}, {difference}")
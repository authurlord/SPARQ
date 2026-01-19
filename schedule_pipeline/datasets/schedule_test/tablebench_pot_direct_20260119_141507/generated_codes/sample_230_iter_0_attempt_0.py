import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between total number of students and research funding
correlation = df['total number of students'].corr(df['research funding (000)'])
print(f"Final Answer: research funding (000)")
import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between human development index (2000) and population density
correlation = df['human development index (2000)'].corr(df['population density ( / km 2 )'])
print(f"Final Answer: {correlation:.3f}")
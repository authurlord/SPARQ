import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between total usaaf and overseas personnel
correlation = df['total usaaf'].corr(df['overseas'])
print(f"Final Answer: {correlation:.3f}")
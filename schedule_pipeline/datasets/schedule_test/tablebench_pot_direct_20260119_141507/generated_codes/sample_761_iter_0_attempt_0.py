import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'bills originally cosponsored' column and compute the mean
average_cosponsored = df['bills originally cosponsored'].mean()
print(f"Final Answer: {average_cosponsored:.1f}")
import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'bills originally cosponsored' column and compute its mean
mean_cosponsored = df['bills originally cosponsored'].mean()
print(f"Final Answer: {mean_cosponsored:.1f}")
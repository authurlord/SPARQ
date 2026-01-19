import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'bills originally cosponsored' column and compute the mean
original_cosponsored = df['bills originally cosponsored'].astype(float).mean()
print(f"Final Answer: {original_cosponsored:.1f}")
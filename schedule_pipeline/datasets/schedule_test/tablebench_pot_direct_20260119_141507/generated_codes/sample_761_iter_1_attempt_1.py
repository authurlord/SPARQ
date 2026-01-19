import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the mean of 'bills originally cosponsored'
mean_cosponsored = df['bills originally cosponsored'].mean()
print(f"Final Answer: {mean_cosponsored:.1f}")
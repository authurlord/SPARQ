import pandas as pd

df = pd.read_csv('table.csv')

# Calculate correlation between 1990 and destroyed
corr_destroyed = df['1990'].corr(df['destroyed'])

# Calculate correlation between 1990 and damaged
corr_damaged = df['1990'].corr(df['damaged'])

print(f"Final Answer: {corr_destroyed:.3f}, {corr_damaged:.3f}")
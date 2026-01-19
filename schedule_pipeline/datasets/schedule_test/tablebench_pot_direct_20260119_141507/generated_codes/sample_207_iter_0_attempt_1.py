import pandas as pd

df = pd.read_csv('table.csv')

# Calculate correlation between initial aircraft (1990) and destroyed, damaged
correlation_destroyed = df['1990'].corr(df['destroyed'])
correlation_damaged = df['1990'].corr(df['damaged'])

print(f"Correlation between 1990 and destroyed: {correlation_destroyed:.3f}")
print(f"Correlation between 1990 and damaged: {correlation_damaged:.3f}")

# Final Answer: Based on correlation, there is no strong evidence of a causal effect.
Final Answer: no
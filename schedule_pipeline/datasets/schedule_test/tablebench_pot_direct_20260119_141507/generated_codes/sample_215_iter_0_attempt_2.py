import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between episodes and viewers
correlation = df['episodes'].corr(df['viewers (in millions)'])
print(f"Final Answer: {correlation:.2f}")
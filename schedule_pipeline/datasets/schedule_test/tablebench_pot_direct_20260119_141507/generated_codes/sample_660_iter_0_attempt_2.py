import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between 'official itv rating (millions)' and 'share (%)'
correlation = df['official itv rating (millions)'].corr(df['share (%)'])
print(f"Final Answer: {correlation:.3f}")
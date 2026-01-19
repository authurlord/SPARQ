import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'males (%)' to numeric and compute the mean
mean_males = df['males (%)'].astype(float).mean()
print(f"Final Answer: {mean_males:.2f}")
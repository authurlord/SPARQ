import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'males (%)' column to numeric and calculate mean
mean_males = df['males (%)'].astype(float).mean()
print(f"Final Answer: {mean_males:.1f}")
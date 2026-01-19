import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'males (%)' column to numeric and calculate mean
average_males = df['males (%)'].astype(float).mean()
print(f"Final Answer: {average_males:.1f}")
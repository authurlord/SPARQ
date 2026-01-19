import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'males (%)' to numeric and calculate the mean
mean_males_percentage = df['males (%)'].astype(float).mean()
print(f"Final Answer: {mean_males_percentage:.2f}")
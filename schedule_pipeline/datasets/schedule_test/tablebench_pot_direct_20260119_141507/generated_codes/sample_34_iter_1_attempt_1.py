import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'time' column to numeric and calculate the mean
mean_time = df['time'].astype(float).mean()
print(f"Final Answer: {mean_time:.2f}")
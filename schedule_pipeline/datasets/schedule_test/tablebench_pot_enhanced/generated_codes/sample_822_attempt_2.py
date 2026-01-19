import pandas as pd

df = pd.read_csv('table.csv')
# Calculate mean and standard deviation of 'us viewers (millions)'
mean_viewers = df['us viewers (millions)'].mean()
std_viewers = df['us viewers (millions)'].std()

print(f"Final Answer: {mean_viewers:.2f}, {std_viewers:.2f}")
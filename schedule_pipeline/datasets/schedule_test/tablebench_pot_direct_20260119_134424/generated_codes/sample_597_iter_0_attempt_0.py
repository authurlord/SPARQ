import pandas as pd

df = pd.read_csv('table.csv')
# Select the first 5 episodes and calculate the average viewers
average_viewers = df['viewers (millions)'].head(5).mean()
print(f"Final Answer: {average_viewers:.2f}")
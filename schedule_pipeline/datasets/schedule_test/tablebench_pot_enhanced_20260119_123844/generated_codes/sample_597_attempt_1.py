import pandas as pd

df = pd.read_csv('table.csv')
# Select the first 5 episodes and calculate average viewers
first_five_viewers = df['viewers (millions)'].head(5).mean()
print(f"Final Answer: {first_five_viewers:.2f}")
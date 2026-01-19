import pandas as pd

df = pd.read_csv('table.csv')
# Extract viewers (millions) for the first 5 episodes
first_5_viewers = df['viewers (millions)'].head(5)
average_viewers = first_5_viewers.mean()
print(f"Final Answer: {average_viewers:.2f}")
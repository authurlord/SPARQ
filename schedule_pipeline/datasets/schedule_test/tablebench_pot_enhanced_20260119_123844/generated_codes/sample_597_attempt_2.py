import pandas as pd

df = pd.read_csv('table.csv')
# Extract the first 5 episodes' viewers
first_five_viewers = df['viewers (millions)'].head(5)
# Calculate the average
average_viewers = first_five_viewers.mean()
print(f"Final Answer: {average_viewers:.2f}")
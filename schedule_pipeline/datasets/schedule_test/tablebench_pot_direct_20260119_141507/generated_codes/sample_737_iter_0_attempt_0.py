import pandas as pd

df = pd.read_csv('table.csv')
# Extract viewership values for the last three episodes
recent_viewers = df['viewers (in millions)'].tail(3).values
average_viewers = recent_viewers.mean()
print(f"Final Answer: {average_viewers:.2f}")
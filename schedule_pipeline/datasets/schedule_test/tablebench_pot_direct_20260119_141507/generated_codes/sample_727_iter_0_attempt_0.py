import pandas as pd

df = pd.read_csv('table.csv')
# Find the series with the highest average viewers (in millions)
max_viewers_series = df.loc[df['average viewers (millions)'].idxmax(), 'series']
print(f"Final Answer: {max_viewers_series}")
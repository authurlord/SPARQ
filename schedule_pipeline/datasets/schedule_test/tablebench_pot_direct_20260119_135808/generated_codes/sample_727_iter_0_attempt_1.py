import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'average viewers (millions)' to float for proper comparison
df['average viewers (millions)'] = pd.to_numeric(df['average viewers (millions)'])
# Find the series with the highest average viewership
max_viewers_series = df.loc[df['average viewers (millions)'].idxmax(), 'series']
print(f"Final Answer: {max_viewers_series}")
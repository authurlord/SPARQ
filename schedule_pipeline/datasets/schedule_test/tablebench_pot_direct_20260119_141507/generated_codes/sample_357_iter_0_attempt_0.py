import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for seasons from 2004 to 2007 inclusive
filtered_df = df[(df['season'] >= '2004') & (df['season'] <= '2007')]
# Count the number of racing series (each row is one series)
total_series = len(filtered_df)
print(f"Final Answer: {total_series}")
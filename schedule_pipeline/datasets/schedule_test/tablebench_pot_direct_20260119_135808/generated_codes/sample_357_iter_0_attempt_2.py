import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for seasons 2004 to 2007
filtered_df = df[(df['season'] >= '2004') & (df['season'] <= '2007')]
# Count the number of racing series (number of rows)
num_series = len(filtered_df)
print(f"Final Answer: {num_series}")
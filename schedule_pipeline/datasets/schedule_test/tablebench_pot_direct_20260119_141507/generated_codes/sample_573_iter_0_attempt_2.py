import pandas as pd

df = pd.read_csv('table.csv')
# Filter buildings constructed between 1960 and 1980 (inclusive)
filtered_df = df[(df['year'] >= 1960) & (df['year'] <= 1980)]
# Calculate the average number of floors
average_floors = filtered_df['floors'].mean()
print(f"Final Answer: {average_floors:.1f}")
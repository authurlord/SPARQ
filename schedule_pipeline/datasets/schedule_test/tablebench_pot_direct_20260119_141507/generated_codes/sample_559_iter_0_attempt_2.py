import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where year is between 1895 and 1980 inclusive
filtered_df = df[(df['year'] >= 1895) & (df['year'] <= 1980)]
# Remove rows where 'floors' is missing (i.e., '-')
valid_floors = filtered_df['floors'].dropna()
# Calculate the mean of valid floors
average_floors = valid_floors.mean()
print(f"Final Answer: {average_floors:.1f}")
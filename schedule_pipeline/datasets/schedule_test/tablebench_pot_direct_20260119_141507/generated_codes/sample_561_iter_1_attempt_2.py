import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Year is between 1942 and 1946 inclusive
filtered_df = df[(df['Year'] >= 1942) & (df['Year'] <= 1946)]
# Calculate the mean of 'US Chart position' for the filtered rows
avg_position = filtered_df['US Chart position'].mean()
print(f"Final Answer: {avg_position:.1f}")
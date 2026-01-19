import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where episodes > 20
filtered_df = df[df['episodes'] > 20]
# Calculate the average viewers (in millions) for those seasons
average_viewers = filtered_df['viewers (in millions)'].mean()
print(f"Final Answer: {average_viewers:.2f}")
import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years between 1942 and 1946 inclusive
filtered_df = df[(df['Year'].astype(int) >= 1942) & (df['Year'].astype(int) <= 1946)]
# Convert 'US Chart position' to integer (removing any text like '(R&B)')
us_positions = filtered_df['US Chart position'].str.extract(r'(\d+)').fillna(0).astype(int)
# Calculate the average
average_position = us_positions.mean()
print(f"Final Answer: {average_position:.1f}")
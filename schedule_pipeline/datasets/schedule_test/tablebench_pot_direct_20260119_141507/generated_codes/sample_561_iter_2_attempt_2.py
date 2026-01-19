import pandas as pd

df = pd.read_csv('table.csv')
# Convert Year column to integer for proper filtering
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
# Filter rows where Year is between 1942 and 1946 (inclusive)
filtered_df = df[(df['Year'] >= 1942) & (df['Year'] <= 1946)]
# Calculate the average US Chart position for the filtered rows
avg_position = filtered_df['US Chart position'].mean()
print(f"Final Answer: {avg_position:.1f}")
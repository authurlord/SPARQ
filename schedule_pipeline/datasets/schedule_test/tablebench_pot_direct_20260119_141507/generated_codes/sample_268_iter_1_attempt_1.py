import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'viewers (m)' to numeric, coercing errors to NaN
df['viewers (m)'] = pd.to_numeric(df['viewers (m)'], errors='coerce')

# Filter episodes with viewership >= 10 million and timeslot rank <= 3
filtered_df = df[(df['viewers (m)'] >= 10) & (df['timeslot rank'].astype(int) <= 3)]

# Calculate average rating of filtered episodes
average_rating = filtered_df['rating'].mean()

print(f"Final Answer: {average_rating:.1f}")
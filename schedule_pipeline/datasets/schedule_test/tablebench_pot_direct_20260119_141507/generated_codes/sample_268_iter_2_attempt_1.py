import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'viewers (m)' to numeric, coercing errors to NaN
df['viewers (m)'] = pd.to_numeric(df['viewers (m)'], errors='coerce')

# Filter episodes with viewership >= 10 million and timeslot rank <= 3
filtered_df = df[(df['viewers (m)'] >= 10) & (df['timeslot rank'].astype(int) <= 3)]

# Calculate the average rating of the filtered episodes
if not filtered_df.empty:
    avg_rating = filtered_df['rating'].mean()
else:
    avg_rating = 0

print(f"Final Answer: {avg_rating:.2f}")
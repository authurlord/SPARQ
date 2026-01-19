import pandas as pd

df = pd.read_csv('table.csv')
# Convert necessary columns to numeric, coercing errors to NaN
df['viewers (m)'] = pd.to_numeric(df['viewers (m)'], errors='coerce')
df['timeslot rank'] = pd.to_numeric(df['timeslot rank'], errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Filter episodes with viewers >= 10 million and timeslot rank <= 3
filtered_episodes = df[(df['viewers (m)'] >= 10) & (df['timeslot rank'] <= 3)]

# Calculate average rating
average_rating = filtered_episodes['rating'].mean()

print(f"Final Answer: {average_rating:.2f}")
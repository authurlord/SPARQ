import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'viewers (m)' to float and 'timeslot rank' to int for comparison
df['viewers (m)'] = pd.to_numeric(df['viewers (m)'])
df['timeslot rank'] = pd.to_numeric(df['timeslot rank'])

# Filter episodes with viewership >= 10 million and timeslot rank <= 3
filtered_episodes = df[(df['viewers (m)'] >= 10) & (df['timeslot rank'] <= 3)]

# Calculate average rating
average_rating = filtered_episodes['rating'].mean()

print(f"Final Answer: {average_rating:.2f}")
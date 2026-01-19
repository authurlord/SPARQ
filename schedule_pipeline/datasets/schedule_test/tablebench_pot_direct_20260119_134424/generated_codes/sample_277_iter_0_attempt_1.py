import pandas as pd

df = pd.read_csv('table.csv')

# Calculate median points
median_points = df['points'].median()

# Filter songs with non-English language and points > median
filtered_songs = df[(df['language'] != 'english') & (df['points'] > median_points)]

# Calculate average points for filtered songs
average_points = filtered_songs['points'].mean()

print(f"Final Answer: {average_points:.1f}")
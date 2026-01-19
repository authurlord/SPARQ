import pandas as pd

df = pd.read_csv('table.csv')

# Calculate median points
median_points = df['points'].median()

# Filter songs with non-English language and points > median
non_english_high_points = df[(df['language'] != 'english') & (df['points'] > median_points)]

# Calculate average points for these songs
avg_points = non_english_high_points['points'].mean()

print(f"Final Answer: {avg_points:.1f}")
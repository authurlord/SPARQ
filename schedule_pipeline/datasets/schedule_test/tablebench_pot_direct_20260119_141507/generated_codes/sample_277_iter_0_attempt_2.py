import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Calculate median points of all songs
median_points = df['points'].median()

# Filter songs with non-English language titles and points > median_points
df_filtered = df[
    (df['language'] != 'english') & 
    (df['points'] > median_points)
]

# Calculate average points of the filtered songs
avg_points = df_filtered['points'].mean()

print(f"Final Answer: {avg_points:.1f}")
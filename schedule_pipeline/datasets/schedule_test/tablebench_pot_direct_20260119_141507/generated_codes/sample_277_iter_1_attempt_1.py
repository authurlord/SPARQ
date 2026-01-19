import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'points' column to numeric, coercing errors to NaN and then dropping invalid entries
df['points'] = pd.to_numeric(df['points'], errors='coerce')

# Compute median points of all songs
median_points = df['points'].median()

# Filter songs with non-English language and points greater than median
filtered_df = df[
    (df['language'] != 'english') &
    (df['points'] > median_points)
]

# Calculate average points of the filtered songs
if filtered_df.empty:
    average_points = 0
else:
    average_points = filtered_df['points'].mean()

print(f"Final Answer: {average_points:.1f}")
import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'points' to numeric, coercing errors to NaN and then dropping them
df['points'] = pd.to_numeric(df['points'], errors='coerce')

# Calculate the median points of all songs
median_points = df['points'].median()

# Filter non-English songs with points greater than median
filtered_df = df[
    (df['language'] != 'english') & 
    (df['points'] > median_points)
]

# Compute average points of the filtered songs
if filtered_df.empty:
    average_points = 0
else:
    average_points = filtered_df['points'].mean()

print(f"Final Answer: {average_points:.1f}")
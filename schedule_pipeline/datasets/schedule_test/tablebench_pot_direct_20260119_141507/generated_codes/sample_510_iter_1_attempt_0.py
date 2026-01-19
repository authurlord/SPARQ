import pandas as pd

df = pd.read_csv('table.csv')
# Filter out rows where language is not 'english'
filtered_df = df[df['language'] != 'english']

# Convert 'points' to numeric, coercing errors to NaN, then drop invalid entries
filtered_df['points'] = pd.to_numeric(filtered_df['points'], errors='coerce')
# Drop any rows where points are NaN after conversion
filtered_df = filtered_df.dropna(subset=['points'])

# Calculate the average points
average_points = filtered_df['points'].mean()
print(f"Final Answer: {average_points:.1f}")